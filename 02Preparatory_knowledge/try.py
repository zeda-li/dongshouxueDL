import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)

y = 2 * torch.dot(x, x)
y.backward()
print(x)
print(x.grad)



123123123

456456

789789

101112
