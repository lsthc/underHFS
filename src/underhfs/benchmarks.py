from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from underhfs import tensor
from underhfs.native import status as native_status


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    backend: str
    iterations: int
    seconds: float
    ops_per_second: float
    output_sample: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "backend": self.backend,
            "iterations": self.iterations,
            "seconds": self.seconds,
            "ops_per_second": self.ops_per_second,
            "output_sample": self.output_sample,
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
    start = perf_counter()
    for _ in range(iterations):
        sample = fn()
    elapsed = perf_counter() - start
    return BenchmarkResult(
        name=name,
        backend=backend,
        iterations=iterations,
        seconds=elapsed,
        ops_per_second=iterations / elapsed if elapsed > 0 else float("inf"),
        output_sample=sample,
    )


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
