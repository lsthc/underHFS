from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

from underhfs import tensor
from underhfs.cuda import MemoryPolicy, MemoryTier
from underhfs.native import status as native_status
from underhfs.runtime import MemoryPlanner, OffloadExecutor
from underhfs.tensor import tensor


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    backend: str
    iterations: int
    seconds: float
    ops_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    output_sample: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "backend": self.backend,
            "iterations": self.iterations,
            "seconds": self.seconds,
            "ops_per_second": self.ops_per_second,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "output_sample": self.output_sample,
        }


@dataclass(frozen=True)
class MemoryBenchmarkResult:
    requested_bytes: int
    placements: dict[str, int]
    offload_events: int
    oom_avoided: bool
    bottlenecks: tuple[str, ...]
    prefetch_verified: bool = False

    def to_dict(self) -> dict[str, int | bool | dict[str, int] | list[str]]:
        return {
            "requested_bytes": self.requested_bytes,
            "placements": self.placements,
            "offload_events": self.offload_events,
            "oom_avoided": self.oom_avoided,
            "bottlenecks": list(self.bottlenecks),
            "prefetch_verified": self.prefetch_verified,
        }


def run_microbenchmarks(
    *,
    size: int = 32,
    iterations: int = 20,
    warmup: int = 3,
    include_cuda: bool = True,
) -> list[BenchmarkResult]:
    if size <= 0:
        raise ValueError("size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    results = [
        _bench("add", "cpu", iterations, warmup, lambda: _cpu_add(size)),
        _bench("matmul", "cpu", iterations, warmup, lambda: _cpu_matmul(size)),
    ]
    native = native_status()
    if include_cuda and native.cuda_enabled:
        results.append(_bench("add", "cuda", iterations, warmup, lambda: _cuda_add(size)))
        results.append(_bench("matmul", "cuda", iterations, warmup, lambda: _cuda_matmul(size)))
    return results


def run_memory_benchmark(
    *,
    tensor_bytes: tuple[int, ...] = (64, 128, 256),
    policy: MemoryPolicy | None = None,
    budgets: dict[MemoryTier, int] | None = None,
) -> MemoryBenchmarkResult:
    actual_policy = policy or MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.NVME))
    actual_budgets = budgets or {
        MemoryTier.VRAM: 128,
        MemoryTier.RAM: 256,
        MemoryTier.NVME: 512,
    }
    planner = MemoryPlanner(actual_policy, actual_budgets)
    placements = {tier.value: 0 for tier in actual_policy.tiers}
    offload_events = 0
    for size in tensor_bytes:
        placement = planner.place_bytes(size)
        placements[placement.tier.value] += placement.bytes
        if placement.reason == "oversubscribed-offload" or placement.tier is not actual_policy.tiers[0]:
            offload_events += 1
    snapshot = planner.snapshot()
    bottlenecks = tuple(
        tier
        for tier, state in snapshot.items()
        if state["capacity_bytes"] > 0 and state["used_bytes"] > state["capacity_bytes"]
    )
    prefetch_verified = _verify_offload_prefetch(actual_policy)
    return MemoryBenchmarkResult(
        requested_bytes=sum(tensor_bytes),
        placements=placements,
        offload_events=offload_events,
        oom_avoided=offload_events > 0 and actual_policy.allow_offload,
        bottlenecks=bottlenecks,
        prefetch_verified=prefetch_verified,
    )


def _verify_offload_prefetch(policy: MemoryPolicy) -> bool:
    if MemoryTier.NVME not in policy.tiers:
        return False
    executor = OffloadExecutor(policy)
    handle = executor.offload_tensor(tensor([1.0, 2.0]), MemoryTier.NVME)
    try:
        cached = executor.prefetch_tensor(handle)
        return executor.load_tensor(cached).tolist() == [1.0, 2.0]
    finally:
        executor.release(handle)


def _bench(
    name: str,
    backend: str,
    iterations: int,
    warmup: int,
    fn: Callable[[], float],
) -> BenchmarkResult:
    sample = 0.0
    for _ in range(warmup):
        sample = fn()
    latencies: list[float] = []
    start = perf_counter()
    for _ in range(iterations):
        iteration_start = perf_counter()
        sample = fn()
        latencies.append(perf_counter() - iteration_start)
    elapsed = perf_counter() - start
    return BenchmarkResult(
        name=name,
        backend=backend,
        iterations=iterations,
        seconds=elapsed,
        ops_per_second=iterations / elapsed if elapsed > 0 else float("inf"),
        latency_p50_ms=median(latencies) * 1000.0,
        latency_p95_ms=_percentile(latencies, 0.95) * 1000.0,
        output_sample=sample,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _vector(size: int, offset: float = 0.0) -> list[float]:
    return [float((i % 17) + 1) + offset for i in range(size)]


def _matrix(size: int, offset: float = 0.0) -> list[list[float]]:
    return [
        [float(((row * size + col) % 17) + 1) + offset for col in range(size)]
        for row in range(size)
    ]


def _cpu_add(size: int) -> float:
    out = tensor(_vector(size)) + tensor(_vector(size, 1.0))
    return float(out._storage[0])


def _cpu_matmul(size: int) -> float:
    out = tensor(_matrix(size)) @ tensor(_matrix(size, 1.0))
    return float(out._storage[0])


def _cuda_add(size: int) -> float:
    out = tensor(_vector(size)).cuda() + tensor(_vector(size, 1.0)).cuda()
    return float(out._storage[0])


def _cuda_matmul(size: int) -> float:
    out = tensor(_matrix(size)).cuda() @ tensor(_matrix(size, 1.0)).cuda()
    return float(out._storage[0])
