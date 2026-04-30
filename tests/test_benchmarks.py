from underhfs.benchmarks import run_memory_benchmark, run_microbenchmarks
from underhfs.cuda import MemoryPolicy, MemoryTier


def test_microbenchmarks_return_cpu_results():
    results = run_microbenchmarks(size=2, iterations=1, warmup=1, include_cuda=False)
    payload = [result.to_dict() for result in results]
    assert [item["name"] for item in payload] == ["add", "matmul"]
    assert all(item["backend"] == "cpu" for item in payload)
    assert all(item["ops_per_second"] > 0 for item in payload)
    assert all(item["latency_p50_ms"] >= 0 for item in payload)
    assert all(item["latency_p95_ms"] >= item["latency_p50_ms"] for item in payload)


def test_memory_benchmark_reports_offload_pressure():
    result = run_memory_benchmark(
        tensor_bytes=(16, 16, 16),
        policy=MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.NVME), allow_offload=True),
        budgets={MemoryTier.VRAM: 16, MemoryTier.NVME: 16},
    )
    payload = result.to_dict()
    assert payload["requested_bytes"] == 48
    assert payload["placements"]["vram"] == 16
    assert payload["placements"]["nvme"] == 32
    assert payload["offload_events"] == 2
    assert payload["oom_avoided"] is True
