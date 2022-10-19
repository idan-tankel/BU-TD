import torch
x = torch.randn(2,3, 2)
y = torch.zeros(2,3, 2)
values = torch.tensor([-1,-2])
print(x)
print(y)
print(torch.where(x > values, x, y))