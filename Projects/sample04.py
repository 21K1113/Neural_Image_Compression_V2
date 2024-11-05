# デコーダの入力に、任意の値を加えたもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

train_model = True

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# オートエンコーダの定義（クラスラベルを追加）
class Autoencoder(nn.Module):
    def __init__(self, label_dim):
        super(Autoencoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # ラベルを追加して結合する全結合層
        self.fc1 = nn.Linear(8*7*7 + label_dim, 8*7*7)
        self.fc2 = nn.Linear(8*7*7, 8*7*7)
        
        # デコーダ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # フラット化
        x = torch.cat([x, labels], dim=1)  # ラベルを追加して結合
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 8, 7, 7)  # 元の形に戻す
        x = self.decoder(x)
        return x

# ハイパーパラメータの設定
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
label_dim = 10  # MNISTのクラスラベルは10種類

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
model = Autoencoder(label_dim=label_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if train_model:
    # 訓練ループ
    for epoch in range(num_epochs):
        for data in train_loader:
            img, labels = data
            img = img.to(device)
            labels = F.one_hot(labels, num_classes=label_dim).float().to(device)

            # 順伝播
            output = model(img, labels)
            loss = criterion(output, img)

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # モデルの保存
    torch.save(model.state_dict(), 'autoencoder_with_labels.pth')

else:
    # 訓練済みパラメータのロード
    model.load_state_dict(torch.load('autoencoder_with_labels.pth'))
    model.eval()


# サンプル画像のエンコードとデコード
sample, labels = next(iter(train_loader))
sample = sample.to(device)
labels = F.one_hot(labels, num_classes=label_dim).float().to(device)
with torch.no_grad():
    compressed = model.encoder(sample)
    compressed = compressed.view(compressed.size(0), -1)
    compressed_with_labels = torch.cat([compressed, labels], dim=1)
    compressed_with_labels = F.relu(model.fc1(compressed_with_labels))
    compressed_with_labels = F.relu(model.fc2(compressed_with_labels))
    reconstructed = model.decoder(compressed_with_labels.view(compressed.size(0), 8, 7, 7))

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
