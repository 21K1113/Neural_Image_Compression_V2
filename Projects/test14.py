import torch
from utils import *
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coordinates = torch.tensor([[0,0]]).to(device)
h = 8
w = 8

pe = triangular_positional_encoding_2d(coordinates, h, w, device, dtype=torch.float32)

print(pe)

coordinates = torch.tensor([[0,1,2,3,0,0,0,0], [0,0,0,0,0,1,2,3]]).to(device)

pe2 = triangular_positional_encoding(coordinates, 6, device, dtype=torch.float32)

print(pe2)

# print(pe.shape)
#
# print(torch.max(pe))
#
# print(torch.min(pe))
#
pen = (pe - torch.min(pe)) / (torch.max(pe) - torch.min(pe))
#
# print(pen.max())
#
# print(pen.min())
#
# print(torch.sum(pe[0] - pe[1]))
# print(torch.mean(torch.abs(pe[0] - pe[1])))

for i in range(12):
    image_saved = Image.fromarray((pen[0][i].cpu().numpy() * 255).astype(np.uint8))
    image_saved.save(f'image/test14/{h}_{i}.png')
