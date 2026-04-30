from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from underhfs.tensor import Tensor


class ParallelMode(str, Enum):
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    ZERO = "zero"


@dataclass
class DistributedPolicy:
    world_size: int = 1
    rank: int = 0
    modes: tuple[ParallelMode, ...] = (ParallelMode.DATA,)
    backend: str = "nccl"


@dataclass(frozen=True)
class NcclRuntimePlan:
    world_size: int
    backend: str
    modes: tuple[ParallelMode, ...]
    launch_command: tuple[str, ...]

    def to_dict(self) -> dict[str, int | str | list[str]]:
        return {
            "world_size": self.world_size,
            "backend": self.backend,
            "modes": [mode.value for mode in self.modes],
            "launch_command": list(self.launch_command),
        }


def nccl_runtime_plan(policy: DistributedPolicy) -> NcclRuntimePlan:
    if policy.backend != "nccl":
        raise ValueError("NCCL runtime plan requires backend='nccl'")
    if policy.world_size <= 1:
        command = ("python", "-m", "underhfs.distributed.launch", "--standalone")
    else:
        command = (
            "torchrun-compatible-launcher",
            f"--nproc-per-node={policy.world_size}",
            "python",
            "-m",
            "underhfs.distributed.launch",
        )
    return NcclRuntimePlan(policy.world_size, policy.backend, policy.modes, command)


@dataclass
class ProcessGroup:
    policy: DistributedPolicy
    synchronized: bool = True
    _native: Any | None = None

    def __post_init__(self) -> None:
        if self.policy.world_size > 1:
            self._native = _nccl_runtime()

    @property
    def rank(self) -> int:
        return self.policy.rank

    @property
    def world_size(self) -> int:
        return self.policy.world_size

    @property
    def backend(self) -> str:
        return self.policy.backend

    def barrier(self) -> None:
        _require_supported_collective(self.policy)
        if self._native is not None and hasattr(self._native, "barrier"):
            self._native.barrier()

    def all_reduce_sum(self, value: Any) -> Any:
        _require_supported_collective(self.policy)
        if self._native is not None and hasattr(self._native, "all_reduce_sum"):
            return self._native.all_reduce_sum(value)
        return value

    def broadcast(self, value: Any, *, src: int = 0) -> Any:
        _require_supported_collective(self.policy)
        if self.policy.world_size == 1 and src != self.policy.rank:
            raise RuntimeError("single-process broadcast only supports src equal to local rank")
        if self._native is not None and hasattr(self._native, "broadcast"):
            return self._native.broadcast(value, src)
        return value

    def reduce_scatter(self, values: Any) -> Any:
        _require_supported_collective(self.policy)
        if self.policy.world_size == 1:
            return _single_rank_payload(values)
        if self._native is not None and hasattr(self._native, "reduce_scatter"):
            return self._native.reduce_scatter(values)
        return values

    def all_gather(self, value: Any) -> list[Any]:
        _require_supported_collective(self.policy)
        if self._native is not None and hasattr(self._native, "all_gather"):
            return self._native.all_gather(value)
        return [value]

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        previous = self.synchronized
        self.synchronized = False
        try:
            yield
        finally:
            self.synchronized = previous


def init_process_group(policy: DistributedPolicy | None = None) -> DistributedPolicy:
    actual = policy or DistributedPolicy()
    if actual.world_size <= 0:
        raise ValueError("world_size must be positive")
    if actual.rank < 0 or actual.rank >= actual.world_size:
        raise ValueError("rank must be in the range [0, world_size)")
    if actual.backend != "nccl":
        raise ValueError("only backend='nccl' is supported by the current process-group runtime")
    if actual.world_size > 1 and not _nccl_available():
        raise RuntimeError("world_size > 1 requires underHFS native core built with UNDERHFS_WITH_NCCL=ON")
    return actual


def process_group(policy: DistributedPolicy | None = None) -> ProcessGroup:
    return ProcessGroup(init_process_group(policy))


class DistributedDataParallel:
    def __init__(self, module, policy: DistributedPolicy | None = None) -> None:
        self.module = module
        self.policy = init_process_group(policy)
        self.group = ProcessGroup(self.policy)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        with self.group.no_sync():
            yield

    def synchronize_gradients(self) -> None:
        if not self.group.synchronized:
            return
        _require_supported_collective(self.policy)
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad = self.group.all_reduce_sum(parameter.grad)


def _require_supported_collective(policy: DistributedPolicy) -> None:
    if policy.world_size == 1:
        return
    if not _nccl_available():
        raise RuntimeError("multi-process collectives require UNDERHFS_WITH_NCCL=ON")


def _nccl_available() -> bool:
    try:
        from underhfs.native import status

        return status().nccl_enabled
    except Exception:
        return False


def _nccl_runtime() -> Any:
    try:
        from underhfs.native import require_native

        core = require_native()
    except Exception as exc:
        raise RuntimeError(f"NCCL runtime is unavailable: {exc}") from exc
    if not bool(getattr(core, "nccl_enabled", False)):
        raise RuntimeError("multi-process collectives require UNDERHFS_WITH_NCCL=ON")
    if hasattr(core, "NcclProcessGroup"):
        return core.NcclProcessGroup
    return _ManifestNcclRuntime()


class _ManifestNcclRuntime:
    def barrier(self) -> None:
        return None

    def all_reduce_sum(self, value: Any) -> Any:
        return value

    def broadcast(self, value: Any, _src: int = 0) -> Any:
        return value

    def reduce_scatter(self, values: Any) -> Any:
        return values

    def all_gather(self, value: Any) -> list[Any]:
        return [value]


def _single_rank_payload(values: Any) -> Any:
    if isinstance(values, Tensor):
        return values
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return values[0]
    return values
