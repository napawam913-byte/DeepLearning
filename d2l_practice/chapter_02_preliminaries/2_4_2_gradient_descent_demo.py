import torch

# 目标：让 x 接近 5
# loss = (x - 5)^2
# 当 x = 5 时，loss 最小，为 0

x = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1

for step in range(20):
    loss = (x - 5) ** 2

    loss.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero_()

    print(f"step={step}, x={x.item():.4f}, loss={loss.item():.4f}")