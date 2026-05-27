import torch

A = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print("A =")
print(A)
print("A.shape =", A.shape)

print("A.T =")
print(A.T)
print("A.T.shape =", A.T.shape)

print("A.sum() =", A.sum())
print("A.mean() =", A.mean())

print("按行求和 A.sum(dim=1) =")
print(A.sum(dim=1))
print("A.sum(dim=1).shape =", A.sum(dim=1).shape)

print("按列求和 A.sum(dim=0) =")
print(A.sum(dim=0))
print("A.sum(dim=0).shape =", A.sum(dim=0).shape)

print("按行求均值 A.mean(dim=1) =")
print(A.mean(dim=1))

print("按列求均值 A.mean(dim=0) =")
print(A.mean(dim=0))

B = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("B.shape =", B.shape)

print("B.sum(dim=0).shape =", B.sum(dim=0).shape)
print("B.sum(dim=1).shape =", B.sum(dim=1).shape)
print("B.sum(dim=2).shape =", B.sum(dim=2).shape)