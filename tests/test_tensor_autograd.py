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


def test_view_flatten_and_inplace_share_storage_and_version():
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x.view(4)
    assert y.shape == (4,)
    assert y.strides == (1,)
    y.add_(1.0)
    assert x.tolist() == [[2.0, 3.0], [4.0, 5.0]]
    assert x.version == y.version == 1
    assert x.flatten().tolist() == [2.0, 3.0, 4.0, 5.0]


def test_slice_returns_strided_view_and_backward_scatter():
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x[:, 1:]
    assert y.shape == (2, 2)
    assert y.strides == (3, 1)
    assert y.tolist() == [[2.0, 3.0], [5.0, 6.0]]
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]


def test_non_contiguous_view_requires_reshape_copy():
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = x[:, ::2]
    try:
        y.view(4)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("expected non-contiguous view to fail")
    reshaped = y.reshape(4)
    assert reshaped.tolist() == [1.0, 3.0, 4.0, 6.0]
