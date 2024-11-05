# 学習済みのモデルを読み込み、圧縮表現から画像を解凍する

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

# 圧縮された潜在表現を読み込み
compressed_np = np.load('compressed_image.npy')
compressed = torch.tensor(compressed_np).to(device)

# 潜在表現から再構成された画像を生成
with torch.no_grad():
    reconstructed = model.decoder(compressed)

# 再構成された画像の表示
plt.imshow(reconstructed.cpu().numpy().squeeze(), cmap='gray')
plt.axis('off')
plt.show()
