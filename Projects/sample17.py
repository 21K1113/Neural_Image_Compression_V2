# DeepSDFのフレームワークを使ってNTCを再現する
# ミップレベルごとに学習していない
# 位置エンコーディングとLODがデコーダに入ってない

from PIL import Image
import torchvision.transforms as transforms
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
from collections import defaultdict
import math

image_path = 'data/sancho_512.png'
basename = os.path.basename(image_path)
num_epochs = 10000
image_size = 512
max_mip_level = 9
feature_pyramid_size = image_size // 4
feature_pyramid_channels = 12

num_bits = 8
decoder_input_channels = 60
hidden_layer_channels = 64

crop_size = (256, 256)
num_crops = 8
batch_size = num_crops
ex_crops = False

train_model = True

project_name = "sample17"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_name = f"{project_name}_{device}_{basename}_{num_epochs}_{num_bits}"


class RandomCropDataset(Dataset):
    def __init__(self, dataset, crop_size, num_crops, ex_crops):
        self.dataset = dataset
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.ex_crops = ex_crops
        self.crops, self.coord = self.get_random_crops()

    def get_random_crops(self):
        crops = []
        coord = []
        _, data_height, data_width = self.dataset.shape
        crop_height, crop_width = self.crop_size

        for _ in range(self.num_crops):
            if self.ex_crops:
                sx = (-crop_width + 1, data_width)
                sy = (-crop_height + 1, data_height)
            else:
                sx = (0, data_width - crop_width + 1)
                sy = (0, data_height - crop_height + 1)
            start_x = torch.randint(sx[0], sx[1], (1,)).item()
            start_y = torch.randint(sy[0], sy[1], (1,)).item()
            end_x = start_x + crop_width
            end_y = start_y + crop_height
            if self.ex_crops:
                if end_x > data_width:
                    end_x = data_width
                elif start_x < 0:
                    start_x = 0
                if end_y > data_height:
                    end_y = data_height
                elif start_y < 0:
                    start_y = 0
            crop = self.dataset[:, start_y:end_y, start_x:end_x]
            crop = crop.reshape(3, -1).T
            crops.append(crop)  # チャンネルを最後から最初に移動
            coord.append([start_y, start_x])
        return crops, coord

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        self.crops, self.coord = self.get_random_crops()
        return self.crops[idx], self.coord[idx]


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


# 2のn乗を返す
def return_2_power(base_size):
    count = 0
    x = base_size
    while x != 1:
        x = x // 2
        count += 1
    return count


# 特徴ピラミッドの最大レベルを返す
def return_pyramid_levels(base_size):
    count = return_2_power(base_size)
    return (count + 1) // 2


# 特徴ピラミッドのレベルとミップレベルの関係の辞書を返す
def create_pyramid_mip_levels(image_size, base_size):
    count = return_2_power(image_size)
    feature_pyramid_dict = defaultdict(int)
    levels = return_pyramid_levels(base_size)
    for i in range(count + 1):
        feature_pyramid_dict[i] = (i // 2) - 1
        if feature_pyramid_dict[i] < 0:
            feature_pyramid_dict[i] = 0
        elif feature_pyramid_dict[i] >= levels:
            feature_pyramid_dict[i] = levels - 1
    return feature_pyramid_dict


def create_pyramid(base_size, channels):
    """
    ピラミッド構造の配列を作成する関数
    :param base_size: 最下層の配列のサイズ (base_size, base_size)
    :param levels: ピラミッドのレベル数
    :return: ピラミッド構造の配列（リスト形式）
    """
    levels = return_pyramid_levels(base_size)
    pyramid = []
    for i in range(levels * 2):
        size = base_size // (2 ** i)
        array = torch.randn(channels, size + 1, size + 1, device=device, requires_grad=True)
        pyramid.append(array)
    return pyramid, levels


def positional_encoding(x, y, image_size, num_channels=2):
    # 位置エンコーディングの計算
    pe = torch.zeros((1, num_channels * 2))
    div_term = torch.exp(torch.arange(0, num_channels, 2) * -(math.log(10000.0) / num_channels))
    pe[0, 0:num_channels:2] = torch.sin(x * div_term)
    pe[0, 1:num_channels:2] = torch.cos(x * div_term)
    pe[0, num_channels::2] = torch.sin(y * div_term)
    pe[0, num_channels + 1::2] = torch.cos(y * div_term)
    return pe


def create_g(fp, fl, j, x_indices, y_indices):
    g_0 = fp[fl * 2 + j][:, y_indices, x_indices]
    g_1 = fp[fl * 2 + j][:, y_indices + 1, x_indices]
    g_2 = fp[fl * 2 + j][:, y_indices, x_indices + 1]
    g_3 = fp[fl * 2 + j][:, y_indices + 1, x_indices + 1]
    return g_0, g_1, g_2, g_3


def create_g0_g1(fp, fl, x, y, step_number, x_range, y_range):
    x_g0_index = torch.floor((x_range + x) * step_number).to(torch.int)
    y_g0_index = torch.floor((y_range + y) * step_number).to(torch.int)
    x_g1_tensor = (x_range + x) * step_number / 2
    y_g1_tensor = (y_range + y) * step_number / 2
    x_g1_index = torch.floor(x_g1_tensor).to(torch.int)
    y_g1_index = torch.floor(y_g1_tensor).to(torch.int)
    x_g0_grid, y_g0_grid = torch.meshgrid(x_g0_index, y_g0_index, indexing='ij')
    x_g1_grid, y_g1_grid = torch.meshgrid(x_g1_index, y_g1_index, indexing='ij')
    x_g0_indices, y_g0_indices = x_g0_grid.reshape(-1), y_g0_grid.reshape(-1)
    x_g1_indices, y_g1_indices = x_g1_grid.reshape(-1), y_g1_grid.reshape(-1)
    g0_0, g0_1, g0_2, g0_3 = create_g(fp, fl, 0, x_g0_indices, y_g0_indices)
    g1_0, g1_1, g1_2, g1_3 = create_g(fp, fl, 1, x_g1_indices, y_g1_indices)
    if int(1 // (step_number / 2)) != 1:
        x_g1_k = x_g1_tensor - x_g1_index.to(torch.float)
        y_g1_k = y_g1_tensor - y_g1_index.to(torch.float)
        x_g1_k_grid, y_g1_k_grid = torch.meshgrid(x_g1_k, y_g1_k, indexing='ij')
        x_g1_k_indices, y_g1_k_indices = x_g1_k_grid.reshape(-1), y_g1_k_grid.reshape(-1)
        g1_0 = g1_0 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices)
        g1_1 = g1_1 * (1 - x_g1_k_indices) * y_g1_k_indices
        g1_2 = g1_2 * x_g1_k_indices * (1 - y_g1_k_indices)
        g1_3 = g1_3 * x_g1_k_indices * y_g1_k_indices
    return g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3


def create_decoder_input(fp, coord, num_crops, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    decoder_input = []
    sample_number = pow(2, max(0, 8 - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number).to(device)
    y_range = torch.arange(sample_number).to(device)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3 = create_g0_g1(fp, fl, coord[0][i], coord[1][i],
                                                                      step_number, x_range, y_range)
        decoder_input.append(
            torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3], dim=0)
        )
    decoder_input = torch.cat(decoder_input, dim=1)
    # end = time.perf_counter()
    # print(end - start)
    return decoder_input.T


def finally_decode_input(fp, image_size):
    sample_number = image_size
    step_number = 1 / 4
    x_range = torch.arange(sample_number).to(device)
    y_range = torch.arange(sample_number).to(device)
    g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3 = create_g0_g1(fp, 0, 0, 0,
                                                                  step_number, x_range, y_range)
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3], dim=0)
    return decoder_input.T


# モデルの学習
def train_models(fp):
    # 単一の画像での訓練ループ
    count = 0
    for epoch in range(num_epochs):
        start_epoch_time = time.perf_counter()
        for inputs, coord in dataloader:
            decoder_input = create_decoder_input(fp, coord, num_crops, 0, 0)

            if epoch < num_epochs * 0.95:
                # 量子化誤差を考慮した一様分布ノイズを生成
                noise = (torch.rand_like(decoder_input) - 0.5) / (2 ** num_bits)
                decoder_input_noise = decoder_input + noise
            else:
                decoder_input_noise = quantize_norm(decoder_input, num_bits)

            decoder_output = decoder(decoder_input_noise)
            target = inputs.reshape(-1, 3)
            loss = criterion(decoder_output, target)
            psnr = calculate_psnr(quantize_to_bit(decoder_output), quantize_to_bit(target))

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_epoch_time = time.perf_counter()

            elapsed_time = end_epoch_time - start_epoch_time

            writer.add_scalar('Loss/train_epoch_label', loss.item(), epoch + 1)
            writer.add_scalar('Time/epoch_label', elapsed_time, epoch + 1)
            writer.add_scalar('PSNR/epoch', psnr, epoch + 1)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            if (epoch + 1) % 100000 == 0:
                # エンコーダとデコーダの保存
                torch.save(decoder.state_dict(), f'model/{save_name}_{epoch}_decoder.pth')


# デコード
def decode_image(fp, arc_decoder):
    with torch.no_grad():
        decoder_input = finally_decode_input(fp, image_size)
        decoder_output = arc_decoder(decoder_input)
        print(decoder_output.shape)
    return decoder_output.reshape(image_size, image_size, 3)


# モデルのインスタンス化
decoder = ColorDecoder().to(device)
criterion = nn.MSELoss()
feature_pyramid, feature_pyramid_levels = create_pyramid(feature_pyramid_size, feature_pyramid_channels)
feature_pyramid_mip_levels_dict = create_pyramid_mip_levels(image_size, feature_pyramid_size)
optimizer = optim.Adam(feature_pyramid + list(decoder.parameters()), lr=1e-3)
writer = SummaryWriter(f'log/{save_name}')


# 学習・圧縮・復元の処理
def process_images(train_model, fp):
    if train_model:
        start = time.perf_counter()
        train_models(fp)
        end = time.perf_counter()
        print("学習時間：" + str(end - start))

        compressed_fp = []
        for g in fp:
            quantized_g = quantize_to_bit(g, num_bits)
            compressed_fp.append((quantized_g * (pow(2, num_bits) - 1)).to(torch.uint8))

        # デコーダの保存
        torch.save(decoder.state_dict(), f'model/{save_name}_decoder.pth')
        torch.save(compressed_fp, f'feature_pyramid/{save_name}_feature_pyramid.pth')
    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        decoder.eval()
        fp = torch.load(f'feature_pyramid/{save_name}_feature_pyramid.pth')

    # デコード
    start = time.perf_counter()
    reconstructed = decode_image(fp, decoder)
    end = time.perf_counter()
    print("展開時間：" + str(end - start))

    reconstructed_image = reconstructed.squeeze().cpu().numpy() * 255
    reconstructed_image = reconstructed_image.astype(np.uint8)
    reconstructed_image_saved = Image.fromarray(reconstructed_image)
    reconstructed_image_saved.save(f'image/{save_name}.png')

    return reconstructed_image


image_original = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
image = transform(image_original).to(device)

random_crop_dataset = RandomCropDataset(image, crop_size, num_crops, ex_crops)
dataloader = DataLoader(random_crop_dataset, batch_size=batch_size, shuffle=True)

reconstructed_image = process_images(train_model, feature_pyramid)
psnr = calculate_average_psnr(image.cpu().numpy().squeeze().transpose(1, 2, 0).astype(np.float32),
                              reconstructed_image.astype(np.float32))
print("psnr:", psnr)

# 画像の表示
plt.figure(figsize=(6, 3))

# オリジナル画像
plt.subplot(1, 2, 1)
plt.imshow(image.cpu().numpy().squeeze().transpose(1, 2, 0))
plt.axis('off')
plt.title('Original Image')

# 再構成された画像
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.axis('off')
plt.title('Reconstructed Image')

plt.show()
