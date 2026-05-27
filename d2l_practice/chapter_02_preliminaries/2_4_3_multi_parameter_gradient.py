import torch

# 训练数据：当 input_x = 2 时，目标 target_y = 10
input_x = torch.tensor(2.0)
target_y = torch.tensor(10.0)

# 模型参数：y_pred = w * input_x + b
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1

for step in range(20):
    # 1. 前向计算：根据当前 w 和 b 得到预测值
    y_pred = w * input_x + b

    # 2. 计算 loss：预测值和真实值之间的差距
    loss = (y_pred - target_y) ** 2

    # 3. 反向传播：计算 loss 对 w 和 b 的梯度
    loss.backward()

    # 4. 手动更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 5. 清空梯度，准备下一轮
    w.grad.zero_()
    b.grad.zero_()

    print(
        f"step={step}, "
        f"w={w.item():.4f}, "
        f"b={b.item():.4f}, "
        f"y_pred={y_pred.item():.4f}, "
        f"loss={loss.item():.4f}"
    )