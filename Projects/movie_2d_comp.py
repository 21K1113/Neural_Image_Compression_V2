# 3次元データを無理やり2次元に変換し量子化して学習を行う

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
import ffmpeg
from utils import *
from models import *

image_path = 'data/misty_64_64.avi'
basename = os.path.basename(image_path)
num_epochs = 3200000
image_size = 512

num_bits = 8
hidden_layer_channels = 32
encoder_output_cannels = 16

train_model = False
save_model = True

project_name = "movie_2d"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_name = f"{project_name}_{device}_{basename}_{num_epochs}_{num_bits}"

# フルカラー画像用エンコーダの定義
class ColorEncoder(nn.Module):
    def __init__(self):
        super(ColorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_layer_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer_channels, encoder_output_cannels, 3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# フルカラー画像用デコーダの定義
class ColorDecoder(nn.Module):
    def __init__(self):
        super(ColorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_cannels, hidden_layer_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_layer_channels, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# モデルの学習
def train_models():
    # 単一の画像での訓練ループ
    for epoch in range(num_epochs):
        start_epoch_time = time.perf_counter()
        # 順伝播
        encoder_output = encoder(image)

        if epoch < num_epochs * 0.95:
            # 量子化誤差を考慮した一様分布ノイズを生成
            noise = (torch.rand_like(encoder_output) - 0.5) / (2 ** num_bits)
            decoder_input = encoder_output + noise
        else:
            decoder_input = quantize_norm(encoder_output, num_bits)
        
        decoder_output = decoder(decoder_input)
        loss = criterion(decoder_output, image)
        psnr = calculate_psnr(quantize_from_norm_to_bit(decoder_output), quantize_from_norm_to_bit(image))

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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if (epoch + 1) % 100000 == 0:
            # エンコーダとデコーダの保存
            torch.save(encoder.state_dict(), f'model/{save_name}_{epoch}_encoder.pth')
            torch.save(decoder.state_dict(), f'model/{save_name}_{epoch}_decoder.pth')


# エンコード
def encode_image(image):
    with torch.no_grad():
        # 順伝播
        encoder_output = encoder(image)
        decoder_input = quantize_norm(encoder_output, num_bits)
        compressed = (decoder_input * (pow(2, num_bits) - 1)).to(torch.uint8)
    return compressed

# デコード
def decode_image(decoder_input):
    with torch.no_grad():
        decoder_output = decoder(decoder_input)
        reconstructed = decoder_output
    return reconstructed


# モデルのインスタンス化
encoder = ColorEncoder().to(device)
decoder = ColorDecoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
writer = SummaryWriter(f'log/{save_name}')


# 学習・圧縮・復元の処理
def process_images(images, train_model, save_model):
    if train_model:
        start = time.perf_counter()
        train_models()
        end = time.perf_counter()
        print("学習時間：" + str(end - start))
        
        # エンコーダとデコーダの保存
        torch.save(encoder.state_dict(), f'model/{save_name}_encoder.pth')
        torch.save(decoder.state_dict(), f'model/{save_name}_decoder.pth')

    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        decoder.eval()

    if save_model:
        if not train_model:
            # エンコーダの訓練済みパラメータのロード
            encoder.load_state_dict(torch.load(f'model/{save_name}_encoder.pth'))
            encoder.eval()
            
        # エンコード
        start = time.perf_counter()
        compressed = encode_image(image)
        end = time.perf_counter()
        print("圧縮時間：" + str(end - start))
            
        # 圧縮された潜在表現を保存
        compressed_np = compressed.cpu().numpy()
        np.save(f'comp/{save_name}.npy', compressed_np)

    else:
        # 圧縮された潜在表現を読み込み
        compressed_np = np.load(f'comp/{save_name}.npy')
        compressed = torch.tensor(compressed_np).to(device)

    decoder_input = compressed.to(torch.float32) / (pow(2, num_bits) - 1)

    # デコード
    start = time.perf_counter()
    reconstructed = decode_image(decoder_input)
    end = time.perf_counter()
    print("展開時間：" + str(end - start))

    reconstructed_image = reconstructed.squeeze().permute(1, 2, 0) # (512, 512, 3)
    reconstructed_movie = quantize_from_norm_to_bit(reconstructed_image.view(64, 64, 64, 3)).cpu().numpy()
    reconstructed_movie = reconstructed_movie.astype(np.uint8)
    # reconstructed_image = Image.fromarray(reconstructed_image)

    timelaps(reconstructed_movie, f'image/{save_name}.avi')

    return reconstructed_movie

# image = Image.open(image_path).convert('RGB')

movie = readClip(image_path)

# numpy 配列を torch.Tensor に変換し、サイズを変更
image = torch.tensor(movie).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0  # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
image = image.view(1, 3, image_size, image_size)
image = image.to(device)

reconstructed_movie = process_images(image, train_model, save_model)
psnr = calculate_average_psnr(movie.astype(np.float32), reconstructed_movie.astype(np.float32))
print("psnr:", psnr)

# 画像の表示
plt.figure(figsize=(6, 3))

# オリジナル画像
plt.subplot(1, 2, 1)
plt.imshow(movie[0])
plt.axis('off')
plt.title('Original Image')

# 再構成された画像
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_movie[0])
plt.axis('off')
plt.title('Reconstructed Image')

plt.show()
