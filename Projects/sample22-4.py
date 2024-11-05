# DeepSDFのフレームワークを使ってNTCを再現する
# 学習率を見直す

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
import ffmpeg
from utils import *
from models import *
from fp_def import *
import math
import random

# image_path = 'data/texture_8192.jpg'
image_path = 'data/misty_64_64.avi'
basename = os.path.basename(image_path)

mlp_num_dtype = 32

num_epochs = 1000
uniform_distribution_rate = 0.05
no_mip = True
# image_size = 8192
# max_mip_level = 13
image_size = 64
max_mip_level = 6
image_bits = 8
output_bits = 8
feature_pyramid_size = image_size // 4
feature_pyramid_channels = 12
pe_channels = 6
if no_mip:
    max_mip_level = 0

num_bits = 8
decoder_input_channels = feature_pyramid_channels * 5 + pe_channels * 2 + 1
hidden_layer_channels = 64

num_labels = image_size
embedding_dim = pow(image_size // 4, 2)  # 埋め込みベクトルの次元

crop_mip_level = 5
crop_size = pow(2, crop_mip_level)
num_crops = 8
batch_size = num_crops

train_model = True
show_result = False

if mlp_num_dtype == 16:
    mlp_dtype = torch.float16
elif mlp_num_dtype == 32:
    mlp_dtype = torch.float32
elif mlp_num_dtype == 64:
    mlp_dtype = torch.float64

project_name = "sample22-4"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_name = f"{project_name}_{device}_{basename}_{mlp_num_dtype}_{no_mip}_{num_epochs}_{num_bits}"


def random_crop_dataset(datasets, label_index, crop_size, num_crops, uniform_distribution, dim=2):
    crops = []
    coord = []
    if uniform_distribution:
        lod = random.randint(0, max_mip_level)
    else:
        lod = int(math.floor(-math.log2(random.random()) / 2))
        if lod > max_mip_level:
            lod = max_mip_level

    dataset = datasets[lod]
    data_size = dataset.shape[1]
    re_crop_size = max(1, crop_size // pow(2, lod))
    for _ in range(num_crops):
        s = (0, data_size - re_crop_size + 1)
        start_coord = torch.randint(s[0], s[1], (dim,)).to(device)
        end_coord = start_coord + re_crop_size
        if dim == 2:
            crop = dataset[:, label_index, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
        elif dim == 3:
            crop = dataset[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]]
        crop = crop.reshape(3, -1).T
        crops.append(crop)  # チャンネルを最後から最初に移動
        coord.append(start_coord)
    return torch.stack(crops), torch.stack(coord), lod


# フルカラー画像用デコーダの定義
class ColorDecoder(nn.Module):
    def __init__(self):
        super(ColorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_channels, hidden_layer_channels),
            nn.GELU(),
            nn.Linear(hidden_layer_channels, hidden_layer_channels),
            nn.GELU(),
            nn.Linear(hidden_layer_channels, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


def create_decoder_input(fp, coord, num_crops, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    decoder_input = []
    irregular = False
    sample_number = pow(2, max(0, crop_mip_level - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # print("step_number", step_number)
    # print(mip_level, sample_number, step_number)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number, dtype=mlp_dtype).to(device)
    y_range = torch.arange(sample_number, dtype=mlp_dtype).to(device)
    lod_tensor = torch.ones(1, pow(sample_number, 2), dtype=mlp_dtype).to(device)
    # print(x_range)
    # print(coord)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, coord[i][0], coord[i][1],
                                                                      step_number, x_range, y_range, pe_channels, device, mlp_dtype)
        decoder_input.append(
            torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3, pe, lod_tensor * mip_level], dim=0)
        )
    decoder_input = torch.cat(decoder_input, dim=1)
    # end = time.perf_counter()
    # print(end - start)
    return decoder_input.T


def finally_decode_input(fp, image_size, mip_level, x=0, y=0):
    sample_number = image_size
    fl = feature_pyramid_mip_levels_dict[mip_level]
    step_number = pow(2, mip_level - (fl + 1) * 2)
    x_range = torch.arange(sample_number, dtype=mlp_dtype).to(device)
    y_range = torch.arange(sample_number, dtype=mlp_dtype).to(device)
    lod_tensor = torch.ones(1, sample_number * sample_number, dtype=mlp_dtype).to(device)
    g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, x, y,
                                                                  step_number, x_range, y_range, pe_channels, device, mlp_dtype)
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3, pe, lod_tensor * mip_level], dim=0)
    return decoder_input.T


# モデルの学習
def train_models(fp):
    # 単一の画像での訓練ループ
    count = 0
    accumulator = 0.0
    judge_freeze = True
    for epoch in range(num_epochs):
        start_epoch_time = time.perf_counter()
        accumulator += uniform_distribution_rate
        if accumulator >= 1.0:
            accumulator -= 1.0
            uniform_distribution = True
        else:
            uniform_distribution = False
        loss_add = 0.0
        psnr_add = 0.0
        for label_index in range(num_labels):
            if epoch > num_epochs * 0.95:
                if judge_freeze:
                    fp_freeze(fp)
                    fp = fp_all_quantize(fp, num_bits)
                    judge_freeze = False
            inputs, coord, lod = random_crop_dataset(images, label_index, crop_size, num_crops, uniform_distribution, dim=2)
            # print(coord)
            # print(convert_coordinate_start(device, coord, 8, 8))
            fl = feature_pyramid_mip_levels_dict[lod]
            # print(lod, fl)

            decoder_input = create_decoder_input(fp, coord, num_crops, fl, lod)
            # print(torch.any(torch.isinf(decoder_input)))
            if epoch < num_epochs * 0.95:
                # 量子化誤差を考慮した一様分布ノイズを生成
                noise = (torch.rand_like(decoder_input, dtype=mlp_dtype) - 0.5) / (2 ** num_bits)
                decoder_input_noise = decoder_input + noise
            else:
                # decoder_input_noise = quantize_norm(decoder_input, num_bits)
                decoder_input_noise = decoder_input

            label_emb = label_embedding(torch.tensor([label_index], dtype=torch.long).to(device))

            decoder_output = decoder(torch.cat(decoder_input_noise, label_emb))
            # decoder_output = decoder(decoder_input)
            target = inputs.reshape(-1, 3)
            loss = criterion(decoder_output, target)
            psnr = calculate_psnr(quantize_to_bit(decoder_output), quantize_to_bit(target))

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_add += loss.item()
            psnr_add += psnr

            fp_quantize_clamp(fp, fl, num_bits)

        end_epoch_time = time.perf_counter()

        elapsed_time = end_epoch_time - start_epoch_time

        writer.add_scalar('Loss/train_epoch_label', loss_add / num_labels, epoch + 1)
        writer.add_scalar('Time/epoch_label', elapsed_time, epoch + 1)
        writer.add_scalar('PSNR/epoch', psnr_add / num_labels, epoch + 1)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_add / num_labels:.4f}')

        if (epoch + 1) % 100000 == 0:
            # エンコーダとデコーダの保存
            torch.save(decoder.state_dict(), f'model/{save_name}_{epoch}_decoder.pth')


# デコード
def decode_image(fp, arc_decoder, mip_level, div_size=10):
    with (torch.no_grad()):
        power = max_mip_level - mip_level
        div_slice = pow(2, max(power - div_size, 0))
        div_count = pow(div_slice, 2)
        if div_count == 1:
            decoder_input = finally_decode_input(fp, image_size // pow(2, mip_level), mip_level)
            decoder_output = arc_decoder(decoder_input)
            print(decoder_output.shape)
            return decoder_output.reshape(image_size // pow(2, mip_level), image_size // pow(2, mip_level), 3)
        else:
            result = torch.zeros(image_size // pow(2, mip_level), image_size // pow(2, mip_level), 3, dtype=mlp_dtype)
            for i in range(div_count):
                sample_number = image_size // pow(2, mip_level + max(power - div_size, 0))
                x = i % div_slice
                y = i // div_slice
                decoder_input = finally_decode_input(fp, sample_number, mip_level, sample_number*x, sample_number*y)
                decoder_output = arc_decoder(decoder_input)
                print(decoder_output.shape)
                result[sample_number*x:sample_number*(x+1), sample_number*y:sample_number*(y+1), :] = decoder_output.reshape(sample_number, sample_number, 3)
            return result


# モデルのインスタンス化
decoder = ColorDecoder().to(device).to(mlp_dtype)
label_embedding = nn.Embedding(num_labels, embedding_dim).to(device)
criterion = nn.MSELoss()
feature_pyramid, feature_pyramid_levels = create_pyramid(feature_pyramid_size, feature_pyramid_channels, num_bits, device, mlp_dtype, no_mip)
for fp in feature_pyramid:
    safe_statistics(fp)
feature_pyramid_mip_levels_dict = create_pyramid_mip_levels(image_size, feature_pyramid_size)
# optimizer = optim.Adam(feature_pyramid + list(decoder.parameters()), lr=1e-3)
optimizer = optim.Adam([
    {'params': feature_pyramid, 'lr': 0.01},  # fpの初期学習率
    {'params': decoder.parameters(), 'lr': 0.005},  # decoderの初期学習率
    {'params': label_embedding.parameters(), 'lr': 0.01}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
writer = SummaryWriter(f'../log/{save_name}')


# 学習・圧縮・復元の処理
def process_images(train_model, fp):
    if train_model:
        start = time.perf_counter()
        train_models(fp)
        end = time.perf_counter()
        print("学習時間：" + str(end - start))

        for g in fp:
            safe_statistics(g)

        compressed_fp = fp_savable(fp, num_bits, torch.uint8)

        # デコーダの保存
        torch.save(decoder.state_dict(), f'model/{save_name}_decoder.pth')
        torch.save(compressed_fp, f'feature_pyramid/{save_name}_feature_pyramid.pth')

        fp = fp_all_quantize(fp, num_bits)
    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        decoder.eval()
        compressed_fp = torch.load(f'feature_pyramid/{save_name}_feature_pyramid.pth')
        fp = fp_load(compressed_fp, num_bits, mlp_dtype)

    reconstructed_images = []
    # デコード
    for i in range(max_mip_level + 1):
        start = time.perf_counter()
        reconstructed = decode_image(fp, decoder, i)
        end = time.perf_counter()
        print("展開時間：" + str(end - start))

        reconstructed_image = reconstructed.cpu().numpy() * 255
        reconstructed_image = reconstructed_image.astype(bits2dtype_np(output_bits))
        reconstructed_images.append(reconstructed_image)
        reconstructed_image_saved = Image.fromarray(reconstructed_image)
        reconstructed_image_saved.save(f'image/{save_name}_{i}.png')

    return reconstructed_images


movie_original = readClip(image_path)
images = []
for i in range(max_mip_level + 1):
    # [T, H, W, C] -> [C, T, H, W]
    movie = movie_original.transpose(3, 0, 1, 2)
    image = torch.tensor(movie).float() / 255.0
    image = image.to(device)
    images.append(image)

"""
image_np = images[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
image_np = (image_np[0:128, 0:128, :] * 255).astype(np.uint8)

image_saved = Image.fromarray(image_np)
image_saved.save(f'data/{save_name}.png')
"""
reconstructed_images = process_images(train_model, feature_pyramid)
for i in range(max_mip_level + 1):
    psnr = calculate_psnr(images[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 255,
                          reconstructed_images[i].astype(np.float32))
    print("psnr:", psnr)

if show_result:
    # 画像の表示
    plt.figure(figsize=(6, 3))

    # オリジナル画像
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].cpu().numpy().squeeze().transpose(1, 2, 0))
    plt.axis('off')
    plt.title('Original Image')

    # 再構成された画像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[0])
    plt.axis('off')
    plt.title('Reconstructed Image')

    plt.show()
