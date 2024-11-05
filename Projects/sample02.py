# 学習済みのモデルを読み込んで画像を圧縮し、それを保存する

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# オートエンコーダの定義（訓練時と同じ構造）
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # デコーダ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# モデルのインスタンス化
model = Autoencoder().to(device)

# 訓練済みパラメータのロード
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# データセットの準備（サンプル画像用）
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# サンプル画像のエンコードと保存
sample, _ = next(iter(test_loader))
sample = sample.to(device)
with torch.no_grad():
    compressed = model.encoder(sample)

# 圧縮された潜在表現を保存
compressed_np = compressed.cpu().numpy()
np.save('compressed_image.npy', compressed_np)
