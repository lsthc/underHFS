from underhfs import tensor, zeros


def test_shape_stride_and_inplace_version():
    x = zeros((2, 3))
    assert x.shape == (2, 3)
    assert x.strides == (3, 1)
    assert x.version == 0
    x.add_(1)
    assert x.version == 1
    assert x.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_scalar_autograd():
    x = tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert x.grad is not None
    assert abs(x.grad.item() - 7.0) < 1e-9


def test_matmul_autograd():
    x = tensor([[1.0, 2.0]], requires_grad=True)
    w = tensor([[3.0], [4.0]], requires_grad=True)
    y = (x @ w).sum()
    y.backward()
    assert x.grad is not None
    assert w.grad is not None
    assert x.grad.tolist() == [[3.0, 4.0]]
    assert w.grad.tolist() == [[1.0], [2.0]]
