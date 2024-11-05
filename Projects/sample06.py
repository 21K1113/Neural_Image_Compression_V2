# 画像単体で学習を行う

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os


image_path = 'data/mandrill.png'
num_epochs = 1000
image_size = 512

encoder_output_cannels = 8

train_model = True
save_model = True



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
            nn.ReLU()
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
        encoded = encoder(image)
        output = decoder(encoded)
        loss = criterion(output, image)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # エンコーダとデコーダの保存
    torch.save(encoder.state_dict(), 'model/color_image_encoder.pth')
    torch.save(decoder.state_dict(), 'model/color_image_decoder.pth')

else:
    # 訓練済みパラメータのロード
    model.load_state_dict(torch.load('autoencoder.pth'))
    model.eval()

# エンコード
with torch.no_grad():
    compressed = encoder(image)

if save_model:
    # 圧縮された潜在表現を保存
    compressed_np = compressed.cpu().numpy()
    basename = os.path.basename(image_path)
    np.save('comp/{basename}.npy', compressed_np)

# デコード
with torch.no_grad():
    reconstructed = decoder(compressed)



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
