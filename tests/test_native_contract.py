from underhfs.native import probe, status
from underhfs.tensor import tensor


def test_native_contract_probe_when_available():
    state = status()
    if not state.available:
        assert state.reason
        return
    result = probe()
    assert isinstance(result["cuda_enabled"], bool)
    assert result["add"] == [6.0, 8.0, 10.0, 12.0]
    assert result["matmul"] == [19.0, 22.0, 43.0, 50.0]
    assert result["sum"] == [134.0]
    if result["cuda_enabled"]:
        assert result["cuda_add_f32"] == [4.0, 6.0]
        assert result["cuda_tensor_add_f32"] == [4.0, 6.0]
        assert result["cuda_tensor_mul_f32"] == [3.0, 8.0]
        assert result["cuda_tensor_sum_f32"] == [3.0]
        assert result["cuda_tensor_matmul_f32"] == [19.0, 22.0, 43.0, 50.0]


def test_tensor_uses_native_cpu_fast_path_when_available():
    if not status().available:
        return
    left = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    right = tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    added = left + right
    multiplied = left * right
    product = left @ right
    total = product.sum()
    assert added.backend == "native_cpu"
    assert multiplied.backend == "native_cpu"
    assert product.backend == "native_cpu"
    assert total.backend == "native_cpu"
    assert product.tolist() == [[19.0, 22.0], [43.0, 50.0]]
    total.backward()
    assert left.grad is not None
    assert right.grad is not None


def test_tensor_uses_native_cuda_add_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    left = tensor([1.0, 2.0]).cuda()
    right = tensor([3.0, 4.0]).cuda()
    out = left + right
    assert out.backend == "native_cuda"
    assert str(out.device) == "cuda:0"
    assert out.tolist() == [4.0, 6.0]


def test_tensor_uses_native_cuda_mul_and_sum_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    left = tensor([1.0, 2.0, 3.0]).cuda()
    right = tensor([4.0, 5.0, 6.0]).cuda()
    product = left * right
    total = product.sum()
    assert product.backend == "native_cuda"
    assert total.backend == "native_cuda"
    assert product.tolist() == [4.0, 10.0, 18.0]
    assert total.tolist() == 32.0


def test_tensor_uses_native_cuda_matmul_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    left = tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
    right = tensor([[5.0, 6.0], [7.0, 8.0]]).cuda()
    out = left @ right
    assert out.backend == "native_cuda"
    assert str(out.device) == "cuda:0"
    assert out.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_tensor_uses_native_cuda_rectangular_matmul_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    left = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).cuda()
    right = tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]).cuda()
    out = left @ right
    assert out.backend == "native_cuda"
    assert out.tolist() == [[58.0, 64.0], [139.0, 154.0]]


def test_cuda_scalar_ops_preserve_device_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    out = tensor([1.0, 2.0]).cuda() + 3.0
    assert str(out.device) == "cuda:0"
    assert out.tolist() == [4.0, 5.0]


def test_cuda_matmul_backward_preserves_device_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    left = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).cuda()
    right = tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True).cuda()
    loss = (left @ right).sum()
    assert loss.backend == "native_cuda"
    assert str(loss.device) == "cuda:0"
    loss.backward()
    assert left.grad is not None
    assert right.grad is not None
    assert str(left.grad.device) == "cuda:0"
    assert str(right.grad.device) == "cuda:0"
    assert left.grad.tolist() == [[11.0, 15.0], [11.0, 15.0]]
    assert right.grad.tolist() == [[4.0, 4.0], [6.0, 6.0]]
