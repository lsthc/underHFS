from pathlib import Path

from underhfs import tensor
from underhfs.functional import cross_entropy, mse_loss
from underhfs.nn import Conv2d, Embedding, Linear, SelfAttention
from underhfs.serialization import (
    load_binary_state_dict,
    load_state_dict,
    save_binary_state_dict,
    save_state_dict,
)


def test_mse_loss_backward():
    x = tensor([[1.0, 2.0]], requires_grad=True)
    y = tensor([[0.0, 4.0]])
    loss = mse_loss(x, y)
    loss.backward()
    assert x.grad is not None
    assert x.grad.tolist() == [[1.0, -2.0]]


def test_cross_entropy_backward_shape():
    logits = tensor([[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]], requires_grad=True)
    target = tensor([0, 1])
    loss = cross_entropy(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    for row in logits.grad.tolist():
        assert abs(sum(row)) < 1e-9


def test_embedding_accumulates_weight_grad():
    embedding = Embedding(4, 3)
    out = embedding(tensor([1, 1, 2]))
    out.sum().backward()
    assert embedding.weight.grad is not None
    grad = embedding.weight.grad.tolist()
    assert grad[0] == [0.0, 0.0, 0.0]
    assert grad[1] == [2.0, 2.0, 2.0]
    assert grad[2] == [1.0, 1.0, 1.0]


def test_state_serialization_roundtrip(tmp_path=None):
    path = Path("tmp_underhfs_state.json") if tmp_path is None else tmp_path / "state.json"
    model = Linear(2, 1)
    save_state_dict(path, model.state_dict())
    loaded = load_state_dict(path)
    assert loaded == model.state_dict()
    if tmp_path is None:
        path.unlink()


def test_binary_state_serialization_roundtrip(tmp_path=None):
    path = Path("tmp_underhfs_state.uhfsbin") if tmp_path is None else tmp_path / "state.uhfsbin"
    model = Linear(2, 1)
    save_binary_state_dict(path, model.state_dict())
    loaded = load_binary_state_dict(path)
    assert _nested_close(loaded, model.state_dict())
    assert path.read_bytes().startswith(b"UHFSBIN1")
    if tmp_path is None:
        path.unlink()


def _nested_close(left, right, tol=1e-6):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_nested_close(left[key], right[key], tol) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_nested_close(a, b, tol) for a, b in zip(left, right, strict=True))
    return abs(left - right) <= tol


def test_conv2d_forward_backward_smoke():
    conv = Conv2d(1, 1, kernel_size=2, bias=True)
    conv.weight._storage = [1.0, 2.0, 3.0, 4.0]
    conv.bias._storage = [0.5]
    x = tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 1, 1, 1)
    assert y.item() == 30.5
    y.sum().backward()
    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias is not None and conv.bias.grad is not None
    assert x.grad.tolist() == [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert conv.weight.grad.tolist() == [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert conv.bias.grad.tolist() == [1.0]


def test_softmax_attention_backward_smoke():
    attn = SelfAttention(2)
    x = tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    y = attn(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert any(parameter.grad is not None for parameter in attn.parameters())
