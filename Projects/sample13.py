import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, z, x):
        z_x = torch.cat([z, x], dim=-1)
        h = torch.relu(self.fc1(z_x))
        h = torch.relu(self.fc2(h))
        s = self.fc3(h)
        return s

latent_dim = 64  # 潜在コードの次元
input_dim = 3  # 3Dポイントの次元（x, y, z）
num_shapes = 100  # 形状の数
num_points = 1000  # 各形状のポイント数
delta = 0.1  # clampの範囲

# サンプルデータの生成（ここではランダムデータを使用）
X = [torch.randn(num_points, input_dim) for _ in range(num_shapes)]
S = [torch.randn(num_points, 1) for _ in range(num_shapes)]

# 潜在コードの初期化
latent_codes = [torch.randn(latent_dim, requires_grad=True) for _ in range(num_shapes)]

# デコーダーネットワークの初期化
decoder = Decoder()

# オプティマイザの設定
optimizer = optim.Adam([{'params': decoder.parameters()}] + [{'params': z} for z in latent_codes], lr=1e-3)

# clamp関数の定義
def clamp(x, delta):
    return torch.clamp(x, min=-delta, max=delta)

# 損失関数の定義
def loss_fn(pred, target, delta):
    return torch.abs(clamp(pred, delta) - clamp(target, delta)).mean()

sigma_squared = 0.1

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(num_shapes):
        z_i = latent_codes[i]
        x_i = X[i]
        s_i = S[i]

        optimizer.zero_grad()

        # 予測
        s_pred = decoder(z_i.unsqueeze(0).repeat(x_i.size(0), 1), x_i)

        # 損失計算
        reconstruction_loss = loss_fn(s_pred, s_i, delta)
        regularization_loss = (1 / sigma_squared) * torch.sum(z_i**2)
        loss = reconstruction_loss + regularization_loss

        # バックプロパゲーションと最適化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_shapes}')
