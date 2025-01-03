import torch

x = torch.arange(8)
y = torch.arange(8)
z = torch.arange(8)

print(y)

print((0, (y+1)//2*2, 0), (0, y//2*2+1, 0), (1, (y+1)//2*2, 1), (1, y//2*2+1, 0))

print(torch.stack(((x+1)//2, (y+1)//2*2, (z+1)//2*2)))
print(torch.stack(((x+1)//2, y//2*2+1, z//2*2+1)))
print(torch.stack((x//2, (y+1)//2*2, z//2*2+1)))
print(torch.stack((x//2, y//2*2+1, (z+1)//2*2)))

