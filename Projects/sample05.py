# エンコーダとデコーダを分けたもの

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# エンコーダの定義
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# デコーダの定義
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# ハイパーパラメータの設定
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
encoder = Encoder().to(device)
decoder = Decoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# 訓練ループ
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)

        # 順伝播
        encoded = encoder(img)
        output = decoder(encoded)
        loss = criterion(output, img)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# エンコーダとデコーダの保存
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')

# サンプル画像のエンコードとデコード
sample, _ = next(iter(train_loader))
sample = sample.to(device)
with torch.no_grad():
    compressed = encoder(sample)
    reconstructed = decoder(compressed)

# 画像の表示
import matplotlib.pyplot as plt

# オリジナル画像
plt.figure(figsize=(9, 2))
for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(sample[i].cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')

# 再構成された画像
for i in range(6):
    plt.subplot(2, 6, i + 7)
    plt.imshow(reconstructed[i].cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
