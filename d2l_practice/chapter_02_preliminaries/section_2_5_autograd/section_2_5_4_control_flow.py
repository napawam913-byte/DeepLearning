import torch


def f(a):
    b = a * 2

    while b.norm() < 1000:
        b = b * 2

    if b.sum() > 0:
        c = b
    else:
        c = 100 * b

    return c


def main():
    a = torch.randn(size=(), requires_grad=True)

    d = f(a)

    d.backward()

    print("a =", a)
    print("d = f(a) =", d)
    print("a.grad =", a.grad)
    print("d / a =", d / a)
    print("a.grad == d / a:", a.grad == d / a)


if __name__ == "__main__":
    main()