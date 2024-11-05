# 位置エンコーディングを使って画像単体で量子化して画素ごとに学習を行う

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
import time

image_path = 'data/sancho_512.png'
num_epochs = 20000
num_bits = 8
image_size = 512

encoder_output_cannels = 8
pos_num_channels = 4
decoder_input_size = encoder_output_cannels * 4 + pos_num_channels * 2
hidden_size = 64

train_model = True
save_model = True

project_name = "pixel_pos"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_name = f"{project_name}_{device}_{basename}_{num_epochs}_{num_bits}"

# フルカラー画像用エンコーダの定義
class ColorEncoder(nn.Module):
    def __init__(self):
        super(ColorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, encoder_output_cannels, 3, stride=2, padding=1),
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
            nn.Linear(decoder_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# 量子化関数の定義
def quantize(tensor, num_bits):
    scale = pow(2, num_bits) - 1
    return (torch.round(tensor * scale) / scale)


# 位置エンコーディングの関数
def positional_encoding(x, y, image_size, num_channels=2):
    # 位置エンコーディングの計算
    pe = torch.zeros((1, num_channels * 2))
    div_term = torch.exp(torch.arange(0, num_channels, 2) * -(math.log(10000.0) / num_channels))
    pe[0, 0:num_channels:2] = torch.sin(x * div_term)
    pe[0, 1:num_channels:2] = torch.cos(x * div_term)
    pe[0, num_channels::2] = torch.sin(y * div_term)
    pe[0, num_channels + 1::2] = torch.cos(y * div_term)
    return pe


image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

# モデルのインスタンス化
encoder = ColorEncoder().to(device)
decoder = ColorDecoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

if train_model:
    # 単一の画像での訓練ループ
    for epoch in range(num_epochs):
        # 順伝播
        encoder_output = encoder(image)

        x = random.randrange(image_size)
        y = random.randrange(image_size)
        ex = math.floor(x/4)
        ey = math.floor(y/4)

        flatten = nn.Flatten()
        cut_encoder_output = flatten(encoder_output[:, :, ex:ex+2, ey:ey+2])
        

        if epoch < num_epochs * 0.95:
            # 量子化誤差を考慮した一様分布ノイズを生成
            noise = (torch.rand_like(cut_encoder_output) - 0.5) / (2 ** num_bits)
            decoder_input = cut_encoder_output + noise
        else:
            decoder_input = quantize(cut_encoder_output, num_bits)

        pos_encoding = positional_encoding(x, y, image_size, pos_num_channels).to(device)
        decoder_input = torch.cat([decoder_input, pos_encoding], dim=1)
        
        decoder_output = decoder(decoder_input)
        loss = criterion(decoder_output, image[:, :, x, y])

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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
    with torch.no_grad():
        # 順伝播
        encoder_output = encoder(image)
        decoder_input = quantize(encoder_output, num_bits)
        compressed = (decoder_input * (pow(2, num_bits) - 1)).to(torch.uint8)
        
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
with torch.no_grad():
    reconstructed = torch.empty_like(image)
    for x in range(image_size):
        for y in range(image_size):
            ex = math.floor(x/4)
            ey = math.floor(y/4)

            flatten = nn.Flatten()
            cut_decoder_input = flatten(decoder_input[:, :, ex:ex+2, ey:ey+2])
            pos_encoding = positional_encoding(x, y, image_size, pos_num_channels).to(device)
            pos_decoder_input = torch.cat([cut_decoder_input, pos_encoding], dim=1)
            decoder_output = decoder(pos_decoder_input)
            reconstructed[:, :, x, y] = decoder_output
end = time.perf_counter()
print("解凍時間：" + str(end - start))

reconstructed_image = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy() * 255 # (512, 512, 3)
reconstructed_image = reconstructed_image.astype(np.uint8)
reconstructed_image = Image.fromarray(reconstructed_image)
reconstructed_image.save(f'image/{save_name}.png')

# 画像の表示
plt.figure(figsize=(6, 3))

# オリジナル画像
plt.subplot(1, 2, 1)
plt.imshow(image.cpu().numpy().squeeze().transpose(1, 2, 0))
plt.axis('off')
plt.title('Original Image')

# 再構成された画像
plt.subplot(1, 2, 2)
plt.imshow(reconstructed.cpu().numpy().squeeze().transpose(1, 2, 0))
plt.axis('off')
plt.title('Reconstructed Image')

plt.show()
