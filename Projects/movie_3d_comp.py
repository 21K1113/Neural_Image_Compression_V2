# 3次元データを3次元のまま、量子化して学習を行う

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
encode_from_middle = False
encode_from_middle_epoch = 2999999

if encode_from_middle:
    train_model = False
    save_model = True

project_name = "movie_3d"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_name = f"{project_name}_{device}_{basename}_{num_epochs}_{num_bits}"

# フルカラー画像用エンコーダの定義
class ColorEncoder(nn.Module):
    def __init__(self):
        super(ColorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, hidden_layer_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_layer_channels, encoder_output_cannels, 3, stride=2, padding=1),
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
            nn.ConvTranspose3d(encoder_output_cannels, hidden_layer_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_layer_channels, 3, 3, stride=2, padding=1, output_padding=1),
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
        psnr = calculate_psnr(quantize_to_bit(decoder_output), quantize_to_bit(image))

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
        if encode_from_middle:
            decoder.load_state_dict(torch.load(f'model/{save_name}_{encode_from_middle_epoch}_decoder.pth'))
        else: 
            decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        decoder.eval()

    if save_model:
        if not train_model:
            # エンコーダの訓練済みパラメータのロード
            if encode_from_middle:
                encoder.load_state_dict(torch.load(f'model/{save_name}_{encode_from_middle_epoch}_encoder.pth'))
            else: 
                encoder.load_state_dict(torch.load(f'model/{save_name}_encoder.pth'))
            encoder.eval()
            
        # エンコード
        start = time.perf_counter()
        compressed = encode_image(image)
        end = time.perf_counter()
        print("圧縮時間：" + str(end - start))
            
        # 圧縮された潜在表現を保存
        compressed_np = compressed.cpu().numpy()
        if encode_from_middle:
            np.save(f'comp/{save_name}_{encode_from_middle_epoch}.npy', compressed_np)
        else:
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

    reconstructed_image = quantize_to_bit(reconstructed.squeeze()).permute(1, 2, 3, 0).cpu().numpy() # (64, 64, 64, 3)
    reconstructed_image = reconstructed_image.astype(np.uint8)
    # reconstructed_image = Image.fromarray(reconstructed_image)

    if encode_from_middle:
        timelaps(reconstructed_image, f'image/{save_name}_{encode_from_middle_epoch}.avi')
    else:
        timelaps(reconstructed_image, f'image/{save_name}.avi')

    return reconstructed_image

# image = Image.open(image_path).convert('RGB')

movie = readClip(image_path)

# numpy 配列を torch.Tensor に変換し、サイズを変更
image = torch.tensor(movie).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0  # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
# image = image.view(1, 3, image_size, image_size)
image = image.to(device)

reconstructed_image = process_images(image, train_model, save_model)
psnr = calculate_average_psnr(movie.astype(np.float32), reconstructed_image.astype(np.float32))
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
plt.imshow(reconstructed_image[0])
plt.axis('off')
plt.title('Reconstructed Image')

plt.show()
