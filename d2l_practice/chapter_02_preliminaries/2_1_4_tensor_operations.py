import torch

x = torch.tensor([1.0, 2.0, 4.0, 8.0])
y = torch.tensor([2.0, 2.0, 2.0, 2.0])

print("x =", x)
print("y =", y)

print("x + y =", x + y)
print("x - y =", x - y)
print("x * y =", x * y)
print("x / y =", x / y)
print("x ** y =", x ** y)

print("torch.exp(x) =", torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor(
    [[2.0, 1.0, 4.0, 3.0],
     [1.0, 2.0, 3.0, 4.0],
     [4.0, 3.0, 2.0, 1.0]]
)

print("X =")
print(X)
print("Y =")
print(Y)

print("按行拼接 torch.cat((X, Y), dim=0) =")
print(torch.cat((X, Y), dim=0))

print("按列拼接 torch.cat((X, Y), dim=1) =")
print(torch.cat((X, Y), dim=1))

print("X.sum() =", X.sum())
print("X == Y =")
print(X == Y)
print("X < Y =")
print(X < Y)