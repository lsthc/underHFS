from underhfs import tensor
from underhfs.cuda import MemoryPolicy, MemoryTier, _parse_nvidia_smi_devices
from underhfs.runtime import MemoryPlanner, NetworkOffloadClient, NetworkOffloadServer, OffloadExecutor, planner_from_system
from underhfs.tensor import DType


def test_memory_planner_places_across_tiers():
    planner = MemoryPlanner(
        MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM)),
        budgets={MemoryTier.VRAM: 8, MemoryTier.RAM: 64},
    )
    fp32 = tensor([1.0, 2.0], dtype=DType.FP32)
    larger = tensor([1.0, 2.0, 3.0], dtype=DType.FP32)
    first = planner.place_tensor(fp32)
    second = planner.place_tensor(larger)
    assert first.tier is MemoryTier.VRAM
    assert second.tier is MemoryTier.RAM
    assert planner.snapshot()["vram"]["used_bytes"] == 8


def test_memory_planner_offload_oversubscription():
    planner = MemoryPlanner(
        MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.NVME), allow_offload=True),
        budgets={MemoryTier.VRAM: 0, MemoryTier.NVME: 1},
    )
    placement = planner.place_tensor(tensor([1.0], dtype=DType.FP32))
    assert placement.tier is MemoryTier.NVME
    assert placement.reason == "oversubscribed-offload"


def test_nvidia_smi_device_parser():
    parsed = _parse_nvidia_smi_devices("0, NVIDIA RTX 4050 Laptop GPU, 6141, 4096, 580.97\n")
    assert len(parsed) == 1
    assert parsed[0].index == 0
    assert parsed[0].name == "NVIDIA RTX 4050 Laptop GPU"
    assert parsed[0].memory_total_bytes == 6141 * 1024 * 1024
    assert parsed[0].memory_free_bytes == 4096 * 1024 * 1024
    assert parsed[0].driver_version == "580.97"


def test_planner_from_system_returns_configured_tiers():
    planner = planner_from_system(MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM)))
    snapshot = planner.snapshot()
    assert set(snapshot).issubset({"vram", "ram"})
    assert "vram" in snapshot
    assert "ram" in snapshot


def test_nvme_offload_executor_roundtrip(tmp_path=None):
    root = ".underhfs-offload-test" if tmp_path is None else tmp_path
    executor = OffloadExecutor(MemoryPolicy(scratch_path=str(root)))
    handle = executor.offload_tensor(tensor([[1.0, 2.0]]), MemoryTier.NVME)
    assert handle.sha256
    loaded = executor.load_tensor(handle)
    assert handle.tier is MemoryTier.NVME
    assert loaded.tolist() == [[1.0, 2.0]]
    cached = executor.prefetch_tensor(handle)
    assert cached.cached
    assert executor.cache_info()["prefetched_tensors"] == 1
    assert executor.load_tensor(cached).tolist() == [[1.0, 2.0]]
    executor.release(handle)
    assert executor.cache_info()["prefetched_tensors"] == 0


def test_nvme_offload_executor_rejects_corruption(tmp_path=None):
    root = ".underhfs-offload-corrupt-test" if tmp_path is None else tmp_path
    executor = OffloadExecutor(MemoryPolicy(scratch_path=str(root)))
    handle = executor.offload_tensor(tensor([1.0, 2.0]), MemoryTier.NVME)
    with open(handle.path, "ab") as file:
        file.write(b"corrupt")
    try:
        executor.load_tensor(handle)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted offload payload should fail validation")
    finally:
        executor.release(handle)


def test_network_offload_http_roundtrip():
    server = NetworkOffloadServer(port=0).start()
    try:
        client = NetworkOffloadClient(server.url)
        handle = client.offload_tensor(tensor([[1.0, 2.0]]))
        assert handle.tier is MemoryTier.NETWORK
        assert client.load_tensor(handle).tolist() == [[1.0, 2.0]]
        client.release(handle)
        try:
            client.load_tensor(handle)
        except Exception as exc:
            assert "not found" in str(exc).lower() or "404" in str(exc)
        else:
            raise AssertionError("released network offload handle should not load")
    finally:
        server.close()


def test_network_offload_rejects_corrupted_remote_payload():
    server = NetworkOffloadServer(port=0).start()
    try:
        client = NetworkOffloadClient(server.url)
        handle = client.offload_tensor(tensor([1.0, 2.0]))
        server._store[handle.id]["data"] = [9.0, 9.0]
        try:
            client.load_tensor(handle)
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("corrupted network offload payload should fail validation")
    finally:
        server.close()
