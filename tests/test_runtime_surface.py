import json
from urllib.request import Request, urlopen

from underhfs.compile import CompilePolicy, FusionKind, compile, explain
from underhfs.cuda import MemoryPolicy, MemoryTier, RuntimePolicy
from underhfs.data import DataLoader, TensorDataset
from underhfs.distributed import DistributedDataParallel
from underhfs.native import status
from underhfs.nn import Linear
from underhfs.serve import ServeConfig, serve, serve_http
from underhfs.tensor import DType, tensor


def test_policy_surfaces():
    policy = RuntimePolicy(memory=MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.NVME)))
    assert policy.memory.allow_offload
    assert policy.memory.tiers[-1] is MemoryTier.NVME


def test_compile_decorator_attaches_policy():
    @compile(policy=CompilePolicy(enabled=True))
    def fn(x):
        return (x * x + x).sum()

    assert fn._underhfs_compile_policy.enabled
    out = fn(tensor([1.0, 2.0], requires_grad=True))
    report = fn._underhfs_last_compile_report
    assert out.item() == 8.0
    assert report is not None
    assert report.guards[0].shape == (2,)
    assert any(node.op == "sum" for node in report.graph.nodes)
    assert any(group.kind is FusionKind.ELEMENTWISE for group in report.fusion_groups)


def test_compile_explain_returns_serializable_report():
    def fn(x):
        return (x + x).relu()

    report = explain(fn, tensor([-1.0, 2.0]))
    payload = report.to_dict()
    assert payload["guards"][0]["dtype"] == "fp32"
    assert payload["graph"]["nodes"]


def test_compile_guard_specialization_cache_tracks_hits_and_misses():
    @compile(policy=CompilePolicy(enabled=True, guard_specialization=True))
    def fn(x):
        return (x + x).sum()

    first = fn(tensor([1.0, 2.0]))
    first_report = fn._underhfs_last_compile_report
    second = fn(tensor([3.0, 4.0]))
    second_report = fn._underhfs_last_compile_report
    third = fn(tensor([[1.0], [2.0]]))
    third_report = fn._underhfs_last_compile_report

    assert first.item() == 6.0
    assert second.item() == 14.0
    assert third.item() == 6.0
    assert first_report.cache_hit is False
    assert second_report.cache_hit is True
    assert second_report.cache_info.hits == 1
    assert second_report.cache_info.misses == 1
    assert second_report.cache_info.specializations == 1
    assert third_report.cache_hit is False
    assert third_report.cache_info.specializations == 2


def test_data_ddp_and_python_server_surfaces():
    loader = DataLoader(TensorDataset([1, 2, 3]), batch_size=2)
    assert list(loader) == [[1, 2], [3]]
    ddp = DistributedDataParallel(Linear(1, 1))
    assert ddp.policy.world_size == 1
    server = serve(lambda payload: {"echo": payload})
    assert server.predict("ok") == {"echo": "ok"}


def test_json_http_server_predict_surface():
    server = serve_http(lambda payload: {"echo": payload["value"]}, ServeConfig(port=0)).start()
    try:
        with urlopen(f"{server.url}/health", timeout=2) as response:
            assert json.loads(response.read().decode("utf-8")) == {"status": "ok"}
        request = Request(
            f"{server.url}/predict",
            data=json.dumps({"value": "ok"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert json.loads(response.read().decode("utf-8")) == {"result": {"echo": "ok"}}
    finally:
        server.close()


def test_tensor_to_cpu_dtype_and_cuda_error():
    x = tensor([1.0, 2.0]).to(dtype=DType.FP16)
    assert x.dtype is DType.FP16
    assert str(x.cpu().device) == "cpu"
    if status().cuda_enabled:
        assert str(x.cuda().device) == "cuda:0"
    else:
        try:
            x.cuda()
        except RuntimeError as exc:
            assert "native core is unavailable" in str(exc) or "built without CUDA support" in str(exc)
        else:
            raise AssertionError("cuda() should fail while CUDA backend is unavailable")
