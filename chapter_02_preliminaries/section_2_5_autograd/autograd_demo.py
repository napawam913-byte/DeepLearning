import torch


def simple_scalar_gradient():
    print("\n[2.5.1] scalar output gradient")
    x = torch.arange(4.0, requires_grad=True)
    y = 2 * torch.dot(x, x)

    y.backward()

    print("x =", x)
    print("y = 2 * dot(x, x) =", y.item())
    print("x.grad =", x.grad)
    print("expected 4 * x =", 4 * x)

    x.grad.zero_()
    y = x.sum()
    y.backward()
    print("after zero_(), grad of sum(x) =", x.grad)


def non_scalar_backward():
    print("\n[2.5.2] non-scalar backward")
    x = torch.arange(4.0, requires_grad=True)
    y = x * x

    y.backward(gradient=torch.ones_like(y))

    print("y = x * x =", y)
    print("gradient of sum(x * x) =", x.grad)
    print("expected 2 * x =", 2 * x)


def detach_computation():
    print("\n[2.5.3] detach computation")
    x = torch.arange(4.0, requires_grad=True)
    y = x * x
    u = y.detach()
    z = u * x

    z.sum().backward()

    print("y = x * x =", y)
    print("u = y.detach() =", u)
    print("z = u * x =", z)
    print("x.grad =", x.grad)
    print("x.grad == u:", x.grad == u)

    x.grad.zero_()
    y.sum().backward()
    print("gradient can still flow through y itself:", x.grad)
    print("expected 2 * x =", 2 * x)


def control_flow_gradient():
    print("\n[2.5.4] gradient with Python control flow")

    def f(a):
        b = a * 2
        while b.norm() < 1000:
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c

    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()

    print("a =", a)
    print("d = f(a) =", d)
    print("a.grad =", a.grad)
    print("d / a =", d / a)
    print("a.grad == d / a:", a.grad == d / a)


def main():
    simple_scalar_gradient()
    non_scalar_backward()
    detach_computation()
    control_flow_gradient()


if __name__ == "__main__":
    main()
