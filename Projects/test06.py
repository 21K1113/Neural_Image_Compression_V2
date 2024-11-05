import torch

a = torch.tensor([2,3,4,5,6])
b = torch.tensor([3,2,1,2,3])

print(a[b])

c = torch.tensor([[2,3,4],[5,6,7],[8,9,0]])
d = torch.tensor([2,1,0,1,2])

print(c[d])
print(c[d,d])

e = [a, b]
print(e)

for x in e[0]:
    print(x)

f = torch.tensor([[1,1],[1,2],[1,3],[2,1]])
print(c[f[0],f[1]])

g = torch.arange(1, 2, 2)
print(g)

h = torch.tensor([1,2,3,4,5,6])
i = torch.tensor([2,4])

print(h * i)