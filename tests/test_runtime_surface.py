from underhfs.compile import CompilePolicy, compile
from underhfs.cuda import MemoryPolicy, MemoryTier, RuntimePolicy
from underhfs.data import DataLoader, TensorDataset
from underhfs.distributed import DistributedDataParallel
from underhfs.nn import Linear
from underhfs.serve import serve
from underhfs.tensor import DType, tensor


def test_policy_surfaces():
    policy = RuntimePolicy(memory=MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.NVME)))
    assert policy.memory.allow_offload
    assert policy.memory.tiers[-1] is MemoryTier.NVME


def test_compile_decorator_attaches_policy():
    @compile(policy=CompilePolicy(enabled=True))
    def fn(x):
        return x

    assert fn._underhfs_compile_policy.enabled


def test_data_ddp_and_python_server_surfaces():
    loader = DataLoader(TensorDataset([1, 2, 3]), batch_size=2)
    assert list(loader) == [[1, 2], [3]]
    ddp = DistributedDataParallel(Linear(1, 1))
    assert ddp.policy.world_size == 1
    server = serve(lambda payload: {"echo": payload})
    assert server.predict("ok") == {"echo": "ok"}


def test_tensor_to_cpu_dtype_and_cuda_error():
    x = tensor([1.0, 2.0]).to(dtype=DType.FP16)
    assert x.dtype is DType.FP16
    assert str(x.cpu().device) == "cpu"
    try:
        x.cuda()
    except RuntimeError as exc:
        assert "native core is unavailable" in str(exc) or "built without CUDA support" in str(exc)
    else:
        raise AssertionError("cuda() should fail while CUDA backend is unavailable")
