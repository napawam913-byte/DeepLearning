import torch

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.ones_like(X)

print("X =")
print(X)

print("Y =")
print(Y)

before = id(Y)
Y = X + Y
after = id(Y)

print("Y = X + Y 后：")
print(Y)
print("before id =", before)
print("after id =", after)
print("id 是否相同 =", before == after)

Z = torch.zeros_like(Y)
print("Z 初始 id =", id(Z))

Z[:] = X + Y
print("Z[:] = X + Y 后：")
print(Z)
print("Z 修改后 id =", id(Z))

Y += X
print("Y += X 后：")
print(Y)
print("Y 原地修改后 id =", id(Y))

A = X.numpy()
B = torch.from_numpy(A)

print("A 是 NumPy array：")
print(A)
print(type(A))

print("B 是从 NumPy 转回来的 tensor：")
print(B)
print(type(B))

a = torch.tensor([3.5])
print("单元素 tensor a =", a)
print("float(a) =", float(a))
print("int(a) =", int(a))