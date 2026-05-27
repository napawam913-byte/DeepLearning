import torch


def train_with_lr(learning_rate):
    x = torch.tensor(0.0, requires_grad=True)

    print(f"\nlearning_rate = {learning_rate}")

    for step in range(10):
        loss = (x - 5) ** 2
        loss.backward()

        with torch.no_grad():
            x -= learning_rate * x.grad

        x.grad.zero_()

        print(f"step={step}, x={x.item():.4f}, loss={loss.item():.4f}")


train_with_lr(0.01)
train_with_lr(0.1)
train_with_lr(1.0)