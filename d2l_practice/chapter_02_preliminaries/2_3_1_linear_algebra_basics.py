import torch

# 标量：只有一个数字
scalar = torch.tensor(3.0)
print("scalar =", scalar)
print("scalar.shape =", scalar.shape)
print("scalar.ndim =", scalar.ndim)

# 向量：一排数字
vector = torch.tensor([1.0, 2.0, 3.0])
print("vector =", vector)
print("vector.shape =", vector.shape)
print("vector.ndim =", vector.ndim)

# 矩阵：二维表格
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
print("matrix =")
print(matrix)
print("matrix.shape =", matrix.shape)
print("matrix.ndim =", matrix.ndim)

# 三维张量：一摞矩阵
tensor_3d = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("tensor_3d =")
print(tensor_3d)
print("tensor_3d.shape =", tensor_3d.shape)
print("tensor_3d.ndim =", tensor_3d.ndim)

# 取矩阵中的元素
print("matrix[0, 1] =", matrix[0, 1])

# 取矩阵的一行
print("matrix[0] =", matrix[0])

# 取矩阵的一列
print("matrix[:, 1] =", matrix[:, 1])