import torch

# 一个简单函数：y = x^2
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

print("x =", x)
print("y = x^2 =", y)

y.backward()

print("x.grad =", x.grad)

# 再看另一个函数：y = 3x^2 + 2x + 1
x2 = torch.tensor(3.0, requires_grad=True)

y2 = 3 * x2 ** 2 + 2 * x2 + 1

print("x2 =", x2)
print("y2 =", y2)

y2.backward()

print("x2.grad =", x2.grad)