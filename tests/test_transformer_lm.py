from underhfs import tensor
from underhfs.functional import cross_entropy
from underhfs.nn import GELU, TransformerLM
from underhfs.optim import SGD


def test_gelu_backward_smoke():
    x = tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = GELU()(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_transformer_lm_training_step_smoke():
    model = TransformerLM(vocab_size=8, max_seq_len=4, features=4, hidden_features=8, layers=1)
    opt = SGD(model.parameters(), lr=1e-3)
    tokens = tensor([1, 2, 3, 4])
    targets = tensor([2, 3, 4, 5])
    logits = model(tokens)
    assert logits.shape == (4, 8)
    loss = cross_entropy(logits, targets)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    before = [parameter.version for parameter in model.parameters()]
    opt.step()
    after = [parameter.version for parameter in model.parameters()]
    assert any(new > old for old, new in zip(before, after, strict=True))
