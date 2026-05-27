import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 0.0])
y = torch.tensor([0.0, 1.0])
z = torch.tensor([2.0, 0.0])
w = torch.tensor([-1.0, 0.0])

print("x =", x)
print("y =", y)
print("z =", z)
print("w =", w)

print("x 的 L2 范数 =", torch.norm(x))
print("z 的 L2 范数 =", torch.norm(z))

print("x · y =", torch.dot(x, y))
print("x · z =", torch.dot(x, z))
print("x · w =", torch.dot(x, w))

print("cos(x, y) =", F.cosine_similarity(x, y, dim=0))
print("cos(x, z) =", F.cosine_similarity(x, z, dim=0))
print("cos(x, w) =", F.cosine_similarity(x, w, dim=0))

x_norm = F.normalize(x, dim=0)
z_norm = F.normalize(z, dim=0)

print("x_norm =", x_norm)
print("z_norm =", z_norm)
print("归一化后 x_norm · z_norm =", torch.dot(x_norm, z_norm))

# 模拟 3 张商品图的 embedding
image_emb = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # 图1：像运动鞋
    [0.0, 1.0, 0.0, 0.0],  # 图2：像手机壳
    [0.0, 0.0, 1.0, 0.0],  # 图3：像T恤
])

# 模拟 3 段商品标题的 embedding
text_emb = torch.tensor([
    [0.9, 0.1, 0.0, 0.0],  # 文1：运动鞋
    [0.1, 0.8, 0.1, 0.0],  # 文2：手机壳
    [0.0, 0.1, 0.9, 0.0],  # 文3：T恤
])

image_emb = F.normalize(image_emb, dim=1)
text_emb = F.normalize(text_emb, dim=1)

similarity_matrix = image_emb @ text_emb.T

print("similarity_matrix =")
print(similarity_matrix)
print("similarity_matrix.shape =", similarity_matrix.shape)

pred_text_index = torch.argmax(similarity_matrix, dim=1)
print("每张图片最匹配的文本索引 =", pred_text_index)