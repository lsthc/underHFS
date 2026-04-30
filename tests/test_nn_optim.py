from underhfs import DType, tensor
from underhfs.nn import Linear, Sequential, ReLU
from underhfs.native import status
from underhfs.optim import AdamW, FusedAdamW, SGD


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


def test_adamw_state_roundtrip_and_validation():
    model = Linear(2, 1)
    opt = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    x = tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    opt.step()
    state = opt.state_dict()
    clone = AdamW(model.parameters(), lr=0.001)
    clone.load_state_dict(state)
    assert clone.state_dict() == state
    try:
        AdamW(model.parameters(), betas=(1.0, 0.9))
    except ValueError as exc:
        assert "betas" in str(exc)
    else:
        raise AssertionError("expected invalid beta to fail")


def test_fused_adamw_updates_parameters():
    model = Linear(2, 1)
    opt = FusedAdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    before = model.weight.tolist()
    loss = model(tensor([[1.0, 2.0]])).sum()
    loss.backward()
    opt.step()
    assert model.weight.tolist() != before


def test_adamw_state_preserves_cuda_dtype_when_available():
    if not status().cuda_enabled:
        return
    param = tensor([1.0, 2.0], dtype=DType.FP16, requires_grad=True).cuda()
    loss = (param * param).sum()
    loss.backward()
    opt = AdamW([param], lr=0.01, weight_decay=0.0)
    opt.step()
    assert opt.m[0].dtype is DType.FP16
    assert str(opt.m[0].device) == "cuda:0"
    assert param.backend == "native_cuda"
