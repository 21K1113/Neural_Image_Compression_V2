# tensorの動作確認

import torch

t = torch.tensor([[[2,4],[3,5]],[[4,6],[5,7]]])

print(t)

print(t.dim())

print(torch.max(t))
