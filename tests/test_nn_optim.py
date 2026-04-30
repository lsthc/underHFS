from underhfs import tensor
from underhfs.nn import Linear, Sequential, ReLU
from underhfs.optim import SGD


def test_module_optimizer_step_changes_parameter_version():
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
    opt = SGD(model.parameters(), lr=0.01)
    before = [parameter.version for parameter in model.parameters()]
    x = tensor([[1.0, -1.0]], requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    after = [parameter.version for parameter in model.parameters()]
    assert any(a > b for a, b in zip(after, before, strict=True))


def test_state_dict_roundtrip():
    model = Linear(2, 1)
    state = model.state_dict()
    clone = Linear(2, 1)
    clone.load_state_dict(state)
    assert clone.state_dict() == state
