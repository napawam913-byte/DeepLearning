import torch

x = torch.arange(4.0)
print("x =", x)

x.requires_grad_(True)
print("x.requires_grad =", x.requires_grad)
print("x.grad =", x.grad)

y = 2 * torch.dot(x, x)
print("y =", y)

y.backward()
print("x.grad =", x.grad)

print("手动计算 4 * x =", 4 * x)

# 清空梯度
x.grad.zero_()
print("清空后 x.grad =", x.grad)

# 另一个函数
y = x.sum()
print("y = x.sum() =", y)

y.backward()
print("x.grad =", x.grad)