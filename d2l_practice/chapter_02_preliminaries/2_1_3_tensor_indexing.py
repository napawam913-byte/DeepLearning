import torch

X = torch.arange(12).reshape(3, 4)

print("X =")
print(X)
print("X.shape =", X.shape)

print("第一行 X[0] =")
print(X[0])

print("最后一行 X[-1] =")
print(X[-1])

print("第 2 行第 3 列 X[1, 2] =")
print(X[1, 2])

print("前 2 行 X[:2] =")
print(X[:2])

print("所有行的第 2 列 X[:, 1] =")
print(X[:, 1])

print("第 1 行到第 2 行，第 2 列到第 3 列 X[0:2, 1:3] =")
print(X[0:2, 1:3])

X[0, 0] = 99
print("修改 X[0, 0] = 99 后：")
print(X)

X[:2, :] = 12
print("把前 2 行全部改成 12 后：")
print(X)