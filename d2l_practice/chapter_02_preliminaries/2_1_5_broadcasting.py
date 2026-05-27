import torch

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)

print("a =")
print(a)
print("a.shape =", a.shape)

print("b =")
print(b)
print("b.shape =", b.shape)

print("a + b =")
print(a + b)
print("(a + b).shape =", (a + b).shape)

scores = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

bias = torch.tensor([10.0, 20.0, 30.0])

print("scores =")
print(scores)
print("scores.shape =", scores.shape)

print("bias =")
print(bias)
print("bias.shape =", bias.shape)

print("scores + bias =")
print(scores + bias)
print("(scores + bias).shape =", (scores + bias).shape)