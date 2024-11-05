# DeepSDFのフレームワークを使ってNTCを再現する
# 位置エンコーディングまだ、平坦化で3次元拡張を目指す

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
from utils import *
from models import *
from fp_def import *
import math
import random
import datetime
import sys
from var import *

# image_path = 'data/misty_64_64.avi'
image_path = 'data/Multilayer_para3_64.npy'
project_name = "sample22-2"
# image_dtype = "movie"
image_dtype = "ndarray"

# 全体のbit数
mlp_num_dtype = 32

num_epochs = 1000                   # 学習回数
uniform_distribution_rate = 0.05    # 一様分布からサンプリングする割合
no_mip = True                       # mipmapを生成しない
image_3d_size = 64                  # 平坦化時のみ使用する
image_size = 512                    # 入力画像サイズ
max_mip_level = 9                   # 入力画像サイズのミップレベル
image_bits = 8                      # 入力画像のbit数（不完全）
output_bits = 8                     # 出力画像のbit数（不完全）
feature_pyramid_channels = 12       # 特徴ピラミッドのチャンネル数
pe_channels = 6                     # 位置エンコーディングの次元数

num_bits = 8                        # 特徴ピラミッドの量子化ビット数
hidden_layer_channels = 64          # デコーダの中間層のノード数

crop_mip_level = 8                  # ランダムクロップのクロップサイズ
num_crops = 8                       # ランダムクロップの数（＝バッチ数）

train_model = True                  # 学習するかどうか（Falseのとき、学習済みモデルをロードする）
show_result = False                 # 結果をmatplotlibかなんかで表示
print_log = True                    # 一定間隔でlogをprintする
print_psnr = True                   # 一定間隔でpsnrをprintする（logとpsnrをprintしないとき、学習途中のprintがなくなる）

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# コマンドラインから変数を取得し値を更新
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        for var in over_write_variable_dict.keys():
            if arg.startswith(var + "="):
                value = judge_value(arg, over_write_variable_dict[var], var)
                exec(f"{var} = {value}", globals())

basename = os.path.basename(image_path)
feature_pyramid_size = image_size // 4      # 特徴ピラミッドのサイズ
if no_mip:
    max_mip_level = 0
decoder_input_channels = feature_pyramid_channels * 5 + pe_channels * 2 + 1
crop_size = pow(2, crop_mip_level)
batch_size = num_crops
mlp_dtype = bits2dtype_torch(mlp_num_dtype, "float")

save_name = f"{project_name}_{device}_{basename}_{mlp_num_dtype}_{no_mip}_{num_epochs}_{num_bits}"

printlog_path = make_filename_by_seq("./printlog", f"{save_name}.txt")
print_(datetime.datetime.now(), printlog_path)

for var in over_write_variable_dict.keys():
    print_(f"{var} : {eval(var)}", printlog_path)


def random_crop_dataset(datasets, crop_size, num_crops, uniform_distribution, dim=2):
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
            crop = dataset[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
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
    sample_number = pow(2, max(0, 8 - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # print("step_number", step_number)
    # print(mip_level, sample_number, step_number)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number).to(device)
    y_range = torch.arange(sample_number).to(device)
    lod_tensor = torch.ones(1, sample_number * sample_number).to(device)
    # print(x_range)
    # print(coord)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, coord[i][0], coord[i][1],
                                                                          step_number, x_range, y_range, pe_channels,
                                                                          device, mlp_dtype)
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
    x_range = torch.arange(sample_number).to(device)
    y_range = torch.arange(sample_number).to(device)
    lod_tensor = torch.ones(1, sample_number * sample_number).to(device)
    g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, x, y,
                                                                      step_number, x_range, y_range, pe_channels,
                                                                      device, mlp_dtype)
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3, pe, lod_tensor * mip_level], dim=0)
    return decoder_input.T


# モデルの学習
def train_models(fp):
    # 単一の画像での訓練ループ
    count = 0
    accumulator = 0.0
    judge_freeze = True
    for epoch in range(num_epochs):
        accumulator += uniform_distribution_rate
        if accumulator >= 1.0:
            accumulator -= 1.0
            uniform_distribution = True
        else:
            uniform_distribution = False
        if epoch > num_epochs * 0.95:
            if judge_freeze:
                fp_freeze(fp)
                fp = fp_all_quantize(fp, num_bits)
                judge_freeze = False
        start_epoch_time = time.perf_counter()
        inputs, coord, lod = random_crop_dataset(images, crop_size, num_crops, uniform_distribution, dim=2)
        # print(coord)
        # print(convert_coordinate_start(device, coord, 8, 8))
        fl = feature_pyramid_mip_levels_dict[lod]
        # print(lod, fl)

        decoder_input = create_decoder_input(fp, coord, num_crops, fl, lod)
        # print(torch.any(torch.isinf(decoder_input)))
        if epoch < num_epochs * 0.95:
            # 量子化誤差を考慮した一様分布ノイズを生成
            noise = (torch.rand_like(decoder_input) - 0.5) / (2 ** num_bits)
            decoder_input_noise = decoder_input + noise
        else:
            # decoder_input_noise = quantize_norm(decoder_input, num_bits)
            decoder_input_noise = decoder_input

        decoder_output = decoder(decoder_input_noise)
        # decoder_output = decoder(decoder_input)
        target = inputs.reshape(-1, 3)
        loss = criterion(decoder_output, target)
        psnr = calculate_psnr(quantize_to_bit(decoder_output, output_bits), quantize_to_bit(target, output_bits))

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        fp_quantize_clamp(fp, fl, num_bits)

        end_epoch_time = time.perf_counter()

        elapsed_time = end_epoch_time - start_epoch_time

        writer.add_scalar('Loss/train_epoch_label', loss.item(), epoch + 1)
        writer.add_scalar('Time/epoch_label', elapsed_time, epoch + 1)
        writer.add_scalar('PSNR/epoch', psnr, epoch + 1)

        if (epoch + 1) % 100 == 0:
            if print_psnr:
                reconstructed = decode_image(fp_all_quantize(fp, num_bits), decoder, i, False)
                all_psnr = calculate_psnr(quantize_to_bit(reconstructed, output_bits),
                                          quantize_to_bit(images[0].permute(1, 2, 0), output_bits))
                writer.add_scalar('PSNR/mip0', all_psnr, epoch + 1)
                if print_log:
                    print_(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} PSNR: {all_psnr:.4f}', printlog_path)
                else:
                    print_(f'Epoch [{epoch + 1}/{num_epochs}], PSNR: {all_psnr:.4f}',
                           printlog_path)
            elif print_log:
                print_(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', printlog_path)

        if (epoch + 1) % 100000 == 0:
            # エンコーダとデコーダの保存
            torch.save(decoder.state_dict(), f'model/{save_name}_{epoch}_decoder.pth')


# デコード
def decode_image(fp, arc_decoder, mip_level, pr=True, div_size=10):
    with (torch.no_grad()):
        power = max_mip_level - mip_level
        div_slice = pow(2, max(power - div_size, 0))
        div_count = pow(div_slice, 2)
        if div_count == 1:
            decoder_input = finally_decode_input(fp, image_size // pow(2, mip_level), mip_level)
            decoder_output = arc_decoder(decoder_input)
            if pr:
                print_(decoder_output.shape, printlog_path)
            return decoder_output.reshape(image_size // pow(2, mip_level), image_size // pow(2, mip_level), 3)
        else:
            result = torch.zeros(image_size // pow(2, mip_level), image_size // pow(2, mip_level), 3, dtype=mlp_dtype)
            for i in range(div_count):
                sample_number = image_size // pow(2, mip_level + max(power - div_size, 0))
                x = i % div_slice
                y = i // div_slice
                decoder_input = finally_decode_input(fp, sample_number, mip_level, sample_number * x, sample_number * y)
                decoder_output = arc_decoder(decoder_input)
                if pr:
                    print_(decoder_output.shape, printlog_path)
                result[sample_number * x:sample_number * (x + 1), sample_number * y:sample_number * (y + 1),
                :] = decoder_output.reshape(sample_number, sample_number, 3)
            return result


# モデルのインスタンス化
decoder = ColorDecoder().to(device)
criterion = nn.MSELoss()
feature_pyramid, feature_pyramid_levels = create_pyramid(feature_pyramid_size, feature_pyramid_channels, num_bits,
                                                         device, mlp_dtype, no_mip)
for fp in feature_pyramid:
    safe_statistics(fp, printlog_path)
feature_pyramid_mip_levels_dict = create_pyramid_mip_levels(image_size, feature_pyramid_size)
# optimizer = optim.Adam(feature_pyramid + list(decoder.parameters()), lr=1e-3)
optimizer = optim.Adam([
    {'params': feature_pyramid, 'lr': 0.01},  # fpの初期学習率
    {'params': decoder.parameters(), 'lr': 0.005}  # decoderの初期学習率
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
writer = SummaryWriter(f'../log/{save_name}')


# 学習・圧縮・復元の処理
def process_images(train_model, fp):
    if train_model:
        start = time.perf_counter()
        train_models(fp)
        end = time.perf_counter()
        print_("学習時間：" + str(end - start), printlog_path)

        for g in fp:
            safe_statistics(g, printlog_path)

        compressed_fp = fp_savable(fp, num_bits, bits2dtype_torch(num_bits))

        # デコーダの保存

        torch.save(decoder.state_dict(), f'model/{save_name}_decoder.pth')
        torch.save(decoder.state_dict(), make_filename_by_seq(f'model/{save_name}', f'{save_name}_decoder.pth'))
        torch.save(compressed_fp, f'feature_pyramid/{save_name}_feature_pyramid.pth')
        torch.save(compressed_fp,
                   make_filename_by_seq('feature_pyramid/{save_name}', f'{save_name}_feature_pyramid.pth'))

        fp = fp_all_quantize(fp, num_bits)
    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        decoder.eval()
        compressed_fp = torch.load(f'feature_pyramid/{save_name}_feature_pyramid.pth')
        fp = fp_load(compressed_fp, num_bits, bits2dtype_torch(num_bits))

    reconstructed_images = []
    # デコード
    for i in range(max_mip_level + 1):
        start = time.perf_counter()
        reconstructed = decode_image(fp, decoder, i)
        end = time.perf_counter()
        print_("展開時間：" + str(end - start), printlog_path)

        reconstructed_image = reconstructed.cpu().numpy() * 255
        reconstructed_image = reconstructed_image.astype(bits2dtype_np(output_bits))
        reconstructed_images.append(reconstructed_image)
        reconstructed_image_saved = Image.fromarray(reconstructed_image)
        reconstructed_image_saved.save(f'image/{save_name}_{i}.png')

    size = image_3d_size
    reconstructed_movie = np.zeros((size, size, size, 3), dtype=np.uint8)
    for x in range(size):
        row = x // (image_size // size)  # 行の計算
        col = x % (image_size // size)  # 列の計算
        # print(row * size, (row + 1) * size, col * size, (col + 1) * size)
        reconstructed_movie[x] = reconstructed_images[0][row * size:(row + 1) * size, col * size:(col + 1) * size, :]

    timelaps(reconstructed_movie, make_filename_by_seq(f'image/{save_name}', f'{save_name}_0.avi'))

    return reconstructed_images, reconstructed_movie


if image_dtype == "movie":
    movie_original = readClip(image_path)
elif image_dtype == "ndarray":
    movie_original = np.load(image_path)

movie_original = quantize_np(movie_original / 255.0, image_bits) * 255
images = []
# [T, H, W, C] -> [C, H, W, T]
# movie = movie_original.transpose(3, 1, 2, 0)
movie = np.zeros((image_size, image_size, 3), dtype=np.uint8)
for i in range(image_3d_size):
    row = i // (image_size // image_3d_size)  # 行の計算
    col = i % (image_size // image_3d_size)  # 列の計算
    movie[row * image_3d_size:(row + 1) * image_3d_size, col * image_3d_size:(col + 1) * image_3d_size, :] = \
    movie_original[i]
# [H, W, C] -> [C, H, W]
# movie = movie.transpose(2, 0, 1)
# movie = movie.reshape((3, image_size, image_size))

image_original = Image.fromarray(movie, 'RGB')

images = []
for i in range(max_mip_level + 1):
    transform = transforms.Compose([
        transforms.Resize((image_size // pow(2, i), image_size // pow(2, i))),
        transforms.ToTensor()
    ])
    image = transform(image_original).to(device)
    print(image.shape)
    # image_np = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    # image_np = (image_np * 255).astype(np.uint8)
    #
    # image_saved = Image.fromarray(image_np)
    # image_saved.save(f'data/{save_name}_{i}.png')
    images.append(image)

"""
for i in range(max_mip_level + 1):
    image = np.resize(movie, (3, image_size // pow(2, i), image_size // pow(2, i)))
    print(image.shape)
    image_saved = Image.fromarray(image.transpose(1, 2, 0), 'RGB')
    image_saved.save(f'image/{save_name}_original_{i}.png')
    image = torch.tensor(image).float() / 255.0
    image = image.to(device)
    print(image.shape)
    images.append(image)
"""

reconstructed_images, reconstructed_movie = process_images(train_model, feature_pyramid)
for i in range(max_mip_level + 1):
    psnr = calculate_psnr(images[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 255,
                          reconstructed_images[i].astype(np.float32))
    print_(f"psnr: {psnr}", printlog_path)

for i in range(max_mip_level + 1):
    save_result_to_csv(reconstructed_movie, make_filename_by_seq(f'LUT/{save_name}', f'{save_name}_{i}.csv'))

"""
image_original = Image.open(image_path).convert('RGB')
images = []
for i in range(max_mip_level + 1):
    transform = transforms.Compose([
        transforms.Resize((image_size // pow(2, i), image_size // pow(2, i))),
        transforms.ToTensor()
    ])
    image = transform(image_original).to(device)
    print(image.shape)
    # image_np = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    # image_np = (image_np * 255).astype(np.uint8)
    #
    # image_saved = Image.fromarray(image_np)
    # image_saved.save(f'data/{save_name}_{i}.png')
    images.append(image)

reconstructed_images = process_images(train_model, feature_pyramid)
for i in range(max_mip_level + 1):
    psnr = calculate_psnr(images[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 255,
                          reconstructed_images[i].astype(np.float32))
    print("psnr:", psnr)
"""

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

print_(datetime.datetime.now(), printlog_path)
