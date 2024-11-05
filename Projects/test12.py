import torch

num_bits = 2

q_min = -(pow(2, num_bits) - 1) / pow(2, num_bits + 1)
q_max = 1 / 2

print(q_min, q_max)

a = torch.linspace(q_min, q_max, steps=64)

print(a)

b = a * pow(2, num_bits)

print(b)

c = torch.floor(b + 0.5)

print(c)

d = torch.round(b)

print(d)

e = c + pow(2, num_bits-1) - 1

print(e)

f = e.to(torch.uint8)

print(f)

g = b + pow(2, num_bits-1) - 1

print(g)

h = torch.floor(g + 0.5)

print(h)

i = g.to(torch.uint8)

print(i)

print(a.dtype)

j = a.to(torch.float16)

print(j)