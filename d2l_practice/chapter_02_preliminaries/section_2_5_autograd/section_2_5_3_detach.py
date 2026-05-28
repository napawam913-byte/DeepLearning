import torch


def main():
    x = torch.arange(4.0, requires_grad=True)

    y = x * x
    u = y.detach()
    z = u * x

    z.sum().backward()

    print("x =", x)
    print("y = x * x =", y)
    print("u = y.detach() =", u)
    print("z = u * x =", z)
    print("x.grad =", x.grad)
    print("x.grad == u:", x.grad == u)

    x.grad.zero_()

    y.sum().backward()

    print("after y.sum().backward():")
    print("x.grad =", x.grad)
    print("x.grad == 2 * x:", x.grad == 2 * x)


if __name__ == "__main__":
    main()