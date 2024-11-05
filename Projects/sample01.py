# 画像を圧縮して解凍する

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# オートエンコーダの定義
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (N, 1, 28, 28) -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (N, 16, 14, 14) -> (N, 8, 7, 7)
            nn.ReLU()
        )
        # デコーダ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # (N, 8, 7, 7) -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (N, 16, 14, 14) -> (N, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
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
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練ループ
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)

        # 順伝播
        output = model(img)
        loss = criterion(output, img)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルの保存
torch.save(model.state_dict(), 'autoencoder.pth')

# サンプル画像のエンコードとデコード
sample, _ = next(iter(train_loader))
sample = sample.to(device)
with torch.no_grad():
    compressed = model.encoder(sample)
    reconstructed = model.decoder(compressed)

# 画像の表示（必要に応じて）
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
