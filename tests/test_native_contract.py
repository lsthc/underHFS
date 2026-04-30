from underhfs.cuda import allocator_stats, empty_cache, stream_stats, synchronize
from underhfs.native import probe, status
from underhfs.tensor import DType, tensor


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
        assert "cuda_fused_adamw_f32" in result
        assert abs(result["cuda_fused_adamw_f32"]["m"][0] - 0.01) < 1e-6
        assert abs(result["cuda_fused_adamw_f32"]["m"][1] - 0.02) < 1e-6
        assert result["cuda_tensor_add_f16"] == [4.0, 6.0]
        assert result["cuda_tensor_mul_f16"] == [3.0, 8.0]
        assert result["cuda_tensor_add_bf16"] == [4.0, 6.0]
        assert result["cuda_tensor_mul_bf16"] == [3.0, 8.0]
        assert result["cuda_allocator"]["allocated_bytes"] > 0
        assert result["cuda_stream"]["non_blocking_streams"] == 1
        assert result["cuda_stream"]["launches"] > 0


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


def test_tensor_uses_native_cuda_fp16_and_bf16_elementwise_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    for dtype in (DType.FP16, DType.BF16):
        left = tensor([1.0, 2.0], dtype=dtype).cuda()
        right = tensor([3.0, 4.0], dtype=dtype).cuda()
        added = left + right
        multiplied = left * right
        assert added.backend == "native_cuda"
        assert multiplied.backend == "native_cuda"
        assert added.dtype is dtype
        assert multiplied.dtype is dtype
        assert added.tolist() == [4.0, 6.0]
        assert multiplied.tolist() == [3.0, 8.0]


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


def test_cuda_allocator_stats_and_empty_cache_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    empty_cache()
    before = allocator_stats()
    out = tensor([1.0, 2.0]).cuda() + tensor([3.0, 4.0]).cuda()
    assert out.tolist() == [4.0, 6.0]
    after = allocator_stats()
    assert after["allocated_bytes"] >= before["allocated_bytes"]
    del out
    empty_cache()
    cleared = allocator_stats()
    assert cleared["cached_bytes"] == 0


def test_cuda_stream_stats_and_synchronize_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    before = stream_stats()
    out = tensor([1.0, 2.0]).cuda() + tensor([3.0, 4.0]).cuda()
    synchronize()
    after = stream_stats()
    assert out.tolist() == [4.0, 6.0]
    assert after["non_blocking_streams"] == 1
    assert after["launches"] > before["launches"]
    assert after["copies"] >= before["copies"]
    assert after["synchronizations"] > before["synchronizations"]


def test_native_cuda_fused_adamw_kernel_when_available():
    state = status()
    if not state.cuda_enabled:
        return
    from underhfs.native import require_native

    core = require_native()
    if not hasattr(core, "cuda_fused_adamw_f32"):
        return
    result = core.cuda_fused_adamw_f32(
        [1.0, 2.0],
        [0.1, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
        0.01,
        0.9,
        0.999,
        1e-8,
        0.0,
        1,
    )
    assert result["param"][0] < 1.0
    assert result["param"][1] < 2.0
    assert result["m"][0] > 0.0
    assert result["v"][1] > 0.0
