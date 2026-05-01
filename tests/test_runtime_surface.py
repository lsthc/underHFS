import json
import base64
import os
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from underhfs.compile import CompilePolicy, FusionKind, compile, explain, lower_to_plan
from underhfs.cuda import MemoryPolicy, MemoryTier, RuntimePolicy, capability_matrix, require_kernel, supports_kernel
from underhfs.data import DataLoader, TensorDataset
from underhfs.distributed import DistributedDataParallel, DistributedPolicy, nccl_runtime_plan, process_group
from underhfs.native import status
from underhfs.nn import Linear
from underhfs.serve import (
    ServeConfig,
    ServingProtocol,
    StreamSourceKind,
    open_stream,
    protocol_capabilities,
    require_protocol,
    serve,
    serve_cpp,
    serve_cpp_manifest,
    serve_grpc,
    serve_grpc_manifest,
    serve_http,
    serve_protocol,
    serve_websocket,
    serve_websocket_loop,
)
from underhfs.tensor import DType, tensor


def test_policy_surfaces():
    policy = RuntimePolicy(memory=MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.NVME)))
    assert policy.memory.allow_offload
    assert policy.memory.tiers[-1] is MemoryTier.NVME
    assert supports_kernel("add", device="cpu", dtype=DType.FP32)
    assert any(item.op == "matmul" for item in capability_matrix())
    try:
        require_kernel("matmul", device="cuda", dtype=DType.INT4)
    except RuntimeError as exc:
        assert "does not have" in str(exc)
    else:
        raise AssertionError("unsupported CUDA int4 matmul should fail explicitly")


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
    assert payload["plan"]["fallback_backend"] == "eager-python"


def test_compile_lower_to_eager_fused_plan():
    def fn(x):
        return (x * x + x).sum()

    report = explain(fn, tensor([1.0, 2.0]))
    plan = lower_to_plan(report.graph, report.fusion_groups)
    assert any(kernel.executable for kernel in plan.kernels)


def test_compile_native_policy_reports_unsupported_fallback_error():
    def fn(x):
        return (x * x + x).sum()

    try:
        explain(fn, tensor([1.0, 2.0]), policy=CompilePolicy(native_required=True, allow_fallback=False))
    except RuntimeError as exc:
        assert "unsupported fusion groups" in str(exc) or "fallback is disabled" in str(exc)
    else:
        raise AssertionError("native compile mode should reject CPU fallback fusion groups")


def test_compile_native_attention_accepts_mixed_2d_shapes():
    from underhfs.compile import GraphIR, FusionGroup

    graph = GraphIR()
    graph.add(name="qk", op="matmul", inputs=("q", "k"), outputs=("qk",), shape=(3, 3), dtype="fp32", device="cuda:0")
    graph.add(name="weights", op="softmax", inputs=("qk",), outputs=("weights",), shape=(3, 3), dtype="fp32", device="cuda:0")
    graph.add(name="out", op="matmul", inputs=("weights", "v"), outputs=("out",), shape=(3, 5), dtype="fp32", device="cuda:0")
    plan = lower_to_plan(
        graph,
        (FusionGroup(FusionKind.ATTENTION, ("qk", "weights", "out")),),
        policy=CompilePolicy(native_required=True, allow_fallback=False),
    )
    assert plan.kernels[0].backend == "native-cuda-attention"


def test_compile_native_attention_kernel_dispatch_when_available():
    if not status().cuda_enabled:
        return
    from underhfs.compile import GraphIR, FusionGroup

    graph = GraphIR()
    graph.add(name="qk", op="matmul", inputs=("q", "k"), outputs=("qk",), shape=(2, 2), dtype="fp32", device="cuda:0")
    graph.add(name="weights", op="softmax", inputs=("qk",), outputs=("weights",), shape=(2, 2), dtype="fp32", device="cuda:0")
    graph.add(name="out", op="matmul", inputs=("weights", "v"), outputs=("out",), shape=(2, 2), dtype="fp32", device="cuda:0")
    plan = lower_to_plan(
        graph,
        (FusionGroup(FusionKind.ATTENTION, ("qk", "weights", "out")),),
        policy=CompilePolicy(native_required=True, allow_fallback=False),
    )
    q = tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
    k = tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
    v = tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
    out = plan.kernels[0].dispatch(q, k, v, scale=1.0)
    assert out.backend == "native_cuda_attention"
    assert str(out.device) == "cuda:0"
    assert out.tolist()[0][0] > 1.0


def test_compile_native_fused_kernel_dispatch_when_available():
    if not status().cuda_enabled:
        return
    x = tensor([1.0, 2.0]).cuda()
    y = tensor([3.0, 4.0]).cuda()
    report = explain(lambda a, b: (a + b) * (a + b), x, y)
    kernel = next(item for item in report.plan.kernels if item.backend == "native-cuda-fused")
    out = kernel.dispatch(x, y, op="add")
    assert out.backend == "native_cuda"
    assert out.tolist() == [4.0, 6.0]


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
    group = process_group(DistributedPolicy())
    assert group.rank == 0
    assert group.world_size == 1
    assert group.backend == "nccl"
    assert group.all_reduce_sum(3) == 3
    assert group.broadcast("ok") == "ok"
    assert group.reduce_scatter(["only-rank"]) == "only-rank"
    assert group.all_gather("ok") == ["ok"]
    assert nccl_runtime_plan(DistributedPolicy(world_size=2)).to_dict()["backend"] == "nccl"
    if not status().nccl_enabled:
        try:
            process_group(DistributedPolicy(world_size=2))
        except RuntimeError as exc:
            assert "UNDERHFS_WITH_NCCL=ON" in str(exc)
        else:
            raise AssertionError("world_size > 1 should require NCCL availability")
    with ddp.no_sync():
        assert ddp.group.synchronized is False
    assert ddp.group.synchronized is True
    server = serve(lambda payload: {"echo": payload})
    assert server.predict("ok") == {"echo": "ok"}
    assert serve_protocol(lambda payload: payload, ServingProtocol.PYTHON).predict("ok") == "ok"
    assert serve_websocket(lambda payload: payload["value"]).predict_frame('{"value":"ok"}') == '{"result": "ok"}'
    assert any(capability.protocol is ServingProtocol.HTTP and capability.available for capability in protocol_capabilities())
    require_protocol(ServingProtocol.GRPC)
    assert serve_grpc_manifest().to_dict()["protocol"] == "grpc"
    assert serve_cpp_manifest().to_dict()["protocol"] == "cpp"
    try:
        grpc_server = serve_grpc(lambda payload: payload)
    except RuntimeError as exc:
        assert "grpcio" in str(exc)
    else:
        assert grpc_server.predict("ok") == "ok"
        grpc_server.close()


def test_grpc_server_starts_when_dependency_is_available():
    try:
        server = serve_grpc(lambda payload: payload["value"], ServeConfig(port=0)).start()
    except RuntimeError as exc:
        assert "grpcio" in str(exc)
        return
    try:
        import grpc

        with grpc.insecure_channel(f"{server.host}:{server.port}") as channel:
            call = channel.unary_unary("/underhfs.grpc.JsonPredictService/Predict")
            response = call(json.dumps({"value": "ok"}).encode("utf-8"), timeout=2)
        assert json.loads(response.decode("utf-8")) == {"result": "ok"}
    finally:
        server.close()


def test_cpp_server_wrapper_uses_json_predict_loop():
    with TemporaryDirectory() as tmp:
        script = Path(tmp) / "underhfs_cpp_serve.py"
        script.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    payload = json.loads(line)\n"
            "    if payload.get('op') == 'exit': break\n"
            "    if payload.get('op') == 'health': print(json.dumps({'status': 'ok'}), flush=True); continue\n"
            "    if payload.get('op') == 'predict': print(json.dumps({'result': payload.get('payload')}), flush=True); continue\n"
            "    print(json.dumps({'error': 'unsupported op'}), flush=True)\n",
            encoding="utf-8",
        )
        server = serve_cpp([sys.executable, str(script)])
        try:
            assert server.request({"op": "health"}) == {"status": "ok"}
            assert server.predict({"value": "ok"}) == {"value": "ok"}
        finally:
            server.close()


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
        request = Request(
            f"{server.url}/v1/predict",
            data=json.dumps({"value": "ok"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert json.loads(response.read().decode("utf-8")) == {"result": {"echo": "ok"}}
    finally:
        server.close()


def test_file_stream_reads_frames(tmp_path=None):
    path = "tmp_underhfs_stream.bin" if tmp_path is None else tmp_path / "stream.bin"
    if tmp_path is None:
        with open(path, "wb") as handle:
            handle.write(b"abcdef")
    else:
        path.write_bytes(b"abcdef")
    frames = list(open_stream(str(path), StreamSourceKind.FILE, chunk_bytes=2))
    assert [frame.data for frame in frames] == [b"ab", b"cd", b"ef"]
    if tmp_path is None:
        import os

        os.unlink(path)


def test_websocket_server_loop_predicts_json_frame():
    server = serve_websocket_loop(lambda payload: payload["value"], ServeConfig(port=0)).start()
    try:
        with socket.create_connection((server.host, server.port), timeout=2) as sock:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            sock.sendall(
                (
                    "GET / HTTP/1.1\r\n"
                    f"Host: {server.host}:{server.port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode("ascii")
            )
            response = sock.recv(4096).decode("ascii", errors="ignore")
            assert "101 Switching Protocols" in response
            sock.sendall(_masked_ws_text('{"value":"ok"}'))
            assert json.loads(_recv_unmasked_ws_text(sock)) == {"result": "ok"}
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


def _masked_ws_text(text: str) -> bytes:
    payload = text.encode("utf-8")
    mask = b"\x01\x02\x03\x04"
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes([0x81, 0x80 | len(payload)]) + mask + masked


def _recv_unmasked_ws_text(sock: socket.socket) -> str:
    header = sock.recv(2)
    length = header[1] & 0x7F
    if length == 126:
        length = int.from_bytes(sock.recv(2), "big")
    elif length == 127:
        length = int.from_bytes(sock.recv(8), "big")
    return sock.recv(length).decode("utf-8")
