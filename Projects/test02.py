# バイリニア補間の検証

import torch
import torch.nn.functional as F
# サンプルテンソルの作成
input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
# バイリニア補間を適用
output_tensor = F.interpolate(input_tensor, size=(4, 4), mode='bilinear', align_corners=False)
# 損失を定義
loss = output_tensor.sum()
# 逆伝播
loss.backward()
# 勾配の表示
print(input_tensor.grad)
print(output_tensor)
