from utils import triangular_positional_encoding
import torch
from PIL import Image
import numpy as np

x = 0
y = 0
channel = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
cml = 8
ml = 2
fl = 0
fp_r = 2
before = True

ml0_sample_number = pow(2, max(0, cml))
sample_number = pow(2, max(0, cml - ml))
step_number = pow(2, ml - fp_r - fl * 2)
x_range = torch.arange(sample_number).to(device)
y_range = torch.arange(sample_number).to(device)
if not before:
    x_tensor = (x_range + x) * ml0_sample_number / sample_number
    y_tensor = (y_range + y) * ml0_sample_number / sample_number
else:
    x_tensor = (x_range + x) * step_number / 2
    y_tensor = (y_range + y) * step_number / 2
x_pe_grid, y_pe_grid = torch.meshgrid(x_tensor, y_tensor, indexing='ij')
x_pe_indices, y_pe_indices = x_pe_grid.reshape(-1), y_pe_grid.reshape(-1)

pe = triangular_positional_encoding(torch.stack([x_pe_indices, y_pe_indices]), channel, device, dtype)

print(pe.shape)

pe_reshaped = pe.view(12, sample_number, sample_number)

print(pe_reshaped.shape)

print("max:", torch.max(pe_reshaped))
print("min:", torch.min(pe_reshaped))

for i in range(12):
    image_saved = Image.fromarray(((pe_reshaped[i].cpu().numpy()+1)/2 * 255).astype(np.uint8)[0:64, 0:64].repeat(2, axis=0).repeat(2, axis=1))
    # image_saved = Image.fromarray(((pe_reshaped[i].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)[0:64, 0:64])
    # image_saved = Image.fromarray(((pe_reshaped[i].cpu().numpy() + 1) / 2 * 255).astype(np.uint8))
    image_saved.save(f'image/test15/{before}_128_{cml}_{ml}_{fp_r}_{i}.png')


