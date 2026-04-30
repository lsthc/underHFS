from underhfs import tensor, zeros
from underhfs.autograd import checkpoint, checkpoint_sequential, jvp


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


def test_forward_mode_jvp_for_elementwise_and_matmul():
    x = tensor([[1.0, 2.0]])
    w = tensor([[3.0], [4.0]])
    dx = tensor([[1.0, 1.0]])
    dw = tensor([[0.5], [0.5]])
    primal, tangent = jvp(lambda a, b: (a @ b).sum(), (x, w), (dx, dw))
    assert primal.item() == 11.0
    assert tangent.item() == 8.5


def test_checkpoint_marks_training_value_and_keeps_backward():
    x = tensor(3.0, requires_grad=True)
    y = checkpoint(lambda value: value * value, x)
    assert y._underhfs_checkpoint["mode"] == "eager-recompute-contract"
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == 6.0


def test_forward_mode_jvp_for_softmax_view_and_slice():
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    dx = tensor([[0.1, 0.2], [0.3, 0.4]])
    primal, tangent = jvp(lambda value: value[:, :].reshape(4).softmax(), (x,), (dx,))
    assert primal.shape == (4,)
    assert tangent.shape == (4,)
    assert abs(sum(tangent.tolist())) < 1e-9


def test_checkpoint_sequential_marks_chunks_and_keeps_backward():
    x = tensor(2.0, requires_grad=True)
    y = checkpoint_sequential([lambda value: value * value, lambda value: value + value], 2, x)
    assert y._underhfs_checkpoint["mode"] == "eager-recompute-contract"
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == 8.0
