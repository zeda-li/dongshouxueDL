import torch

a = torch.arange(12).reshape((3, 4))
print(a)
print(torch.diag(a))
b = torch.arange(12)
print(b)
print(torch.diag(b))

print(torch.cuda.device_count())