import torch

# 1. 向量点积
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

dot = torch.dot(x, y)

print("x =", x)
print("y =", y)
print("torch.dot(x, y) =", dot)

manual_dot = 1 * 4 + 2 * 5 + 3 * 6
print("手动计算点积 =", manual_dot)

# 2. 矩阵和向量相乘
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
v = torch.tensor([1.0, 2.0, 3.0])

print("A =")
print(A)
print("A.shape =", A.shape)

print("v =", v)
print("v.shape =", v.shape)

Av = torch.mv(A, v)
print("torch.mv(A, v) =", Av)
print("Av.shape =", Av.shape)

# 3. 矩阵和矩阵相乘
B = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print("B =")
print(B)
print("B.shape =", B.shape)

AB = torch.mm(A, B)
print("torch.mm(A, B) =")
print(AB)
print("AB.shape =", AB.shape)

# 4. 使用 @ 做矩阵乘法
AB2 = A @ B
print("A @ B =")
print(AB2)

# 5. 模拟分类层：hidden @ W + b
batch_size = 2
hidden_size = 3
num_labels = 4

hidden = torch.randn(batch_size, hidden_size)
W = torch.randn(hidden_size, num_labels)
b = torch.zeros(num_labels)

logits = hidden @ W + b

print("hidden.shape =", hidden.shape)
print("W.shape =", W.shape)
print("b.shape =", b.shape)
print("logits.shape =", logits.shape)
print("logits =")
print(logits)