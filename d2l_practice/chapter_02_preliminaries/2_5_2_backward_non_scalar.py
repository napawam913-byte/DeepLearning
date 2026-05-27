import torch

x = torch.arange(4.0, requires_grad=True)
print("x =", x)

# y 是向量，不是标量
y = x * x
print("y = x * x =", y)
print("y.shape =", y.shape)

# 方法 1：把 y 求和成标量
y_sum = y.sum()
print("y_sum =", y_sum)

y_sum.backward()
print("对 y.sum() backward 后 x.grad =", x.grad)

# 清空梯度
x.grad.zero_()

# 方法 2：对非标量 y 传入 gradient
y = x * x
external_gradient = torch.ones_like(y)

y.backward(gradient=external_gradient)
print("传入 ones_like(y) 后 x.grad =", x.grad)

# 再试一个不同的外部梯度
x.grad.zero_()

y = x * x
external_gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])

y.backward(gradient=external_gradient)
print("传入 [1,2,3,4] 后 x.grad =", x.grad)