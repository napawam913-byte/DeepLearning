import torch

x = torch.arange(12)
print("原始 x =", x)
print("x.shape =", x.shape)

X = x.reshape(3, 4)
print("reshape 后的 X =")
print(X)
print("X.shape =", X.shape)

zeros = torch.zeros((2, 3, 4))
print("zeros.shape =", zeros.shape)

ones = torch.ones((2, 3, 4))
print("ones.shape =", ones.shape)

random_tensor = torch.randn(3, 4)
print("random_tensor =")
print(random_tensor)
print("random_tensor.shape =", random_tensor.shape)

manual_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("manual_tensor =")
print(manual_tensor)
print("manual_tensor.shape =", manual_tensor.shape)