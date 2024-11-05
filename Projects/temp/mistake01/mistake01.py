# 画像単体で量子化して学習を行う
# 正規化するときに-128~128になってしまい、出力が化け物になった図

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os


image_path = 'data/sancho_512.png'
num_epochs = 1000
num_bits = 8
image_size = 512

encoder_output_cannels = 8

train_model = False
save_model = False



# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# フルカラー画像用エンコーダの定義
class ColorEncoder(nn.Module):
    def __init__(self):
        super(ColorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
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
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# 量子化関数の定義
def quantize(tensor, num_bits):
    scale = pow(2, num_bits) - 1
    return (torch.round(tensor * scale) / scale)

    

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

        if epoch < num_epochs * 0.95:
            # 量子化誤差を考慮した一様分布ノイズを生成
            noise = (torch.rand_like(encoder_output) - 0.5) / (2 ** num_bits)
            decoder_input = encoder_output + noise
        else:
            decoder_input = quantize(encoder_output, num_bits)
        
        decoder_output = decoder(decoder_input)
        loss = criterion(decoder_output, image)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # エンコーダとデコーダの保存
    basename = os.path.basename(image_path)
    torch.save(encoder.state_dict(), f'model/{basename}_color_image_encoder.pth')
    torch.save(decoder.state_dict(), f'model/{basename}_color_image_decoder.pth')

else:
    # デコーダの訓練済みパラメータのロード
    basename = os.path.basename(image_path)
    decoder.load_state_dict(torch.load(f'model/{basename}_color_image_decoder.pth'))
    decoder.eval()



if save_model:
    if not train_model:
        # エンコーダの訓練済みパラメータのロード
        basename = os.path.basename(image_path)
        encoder.load_state_dict(torch.load(f'model/{basename}_color_image_encoder.pth'))
        encoder.eval()
        
    # エンコード
    with torch.no_grad():
        # 順伝播
        encoder_output = encoder(image)
        decoder_input = quantize(encoder_output, num_bits)
        print(decoder_input)
        compressed = (decoder_input * (pow(2, num_bits) - 1)).to(torch.int8)
        print(compressed)
        
    # 圧縮された潜在表現を保存
    compressed_np = compressed.cpu().numpy()
    basename = os.path.basename(image_path)
    np.save(f'comp/{basename}.npy', compressed_np)

else:
    # 圧縮された潜在表現を読み込み
    basename = os.path.basename(image_path)
    compressed_np = np.load(f'comp/{basename}.npy')
    compressed = torch.tensor(compressed_np).to(device)
    decoder_input = compressed.to(torch.float32)

# デコード
with torch.no_grad():
    decoder_output = decoder(decoder_input)
    reconstructed = decoder_output
    

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
