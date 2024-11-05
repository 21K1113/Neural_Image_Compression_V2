import torch
import torch.nn as nn
import torch.optim as optim

# 初期のベクトル
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)


# デコーダ (簡単な線形層)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.fc(x)


# デコーダのインスタンスを作成
decoder = Decoder()

# 目標のベクトル
target = torch.tensor([2.0, 2.0, 2.0])

# 損失関数
criterion = nn.MSELoss()

# オプティマイザを最初に定義
print(list(decoder.parameters()) + [a])
print(type(list(decoder.parameters()) + [a]))
optimizer = optim.Adam(list(decoder.parameters()) + [a], lr=0.01)

# 学習ループ
for epoch in range(100):
    optimizer.zero_grad()

    # デコーダに入力
    output = decoder(a[:3])

    # 損失を計算
    loss = criterion(output, target)

    # 勾配を計算
    loss.backward()

    # オプティマイザのステップ
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# 学習後の値
print('学習後のa:', a.detach().numpy())
print('学習後のaの勾配:', a.grad.numpy())
