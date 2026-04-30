from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator


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

    def barrier(self) -> None:
        _validate_single_process(self.policy)

    def all_reduce_sum(self, value: Any) -> Any:
        _validate_single_process(self.policy)
        return value

    def broadcast(self, value: Any, *, src: int = 0) -> Any:
        _validate_single_process(self.policy)
        if src != self.policy.rank:
            raise RuntimeError("single-process broadcast only supports src equal to local rank")
        return value

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
    if actual.world_size > 1:
        raise RuntimeError(
            "multi-process distributed execution is reserved for the NCCL runtime; "
            "use world_size=1 for the current local process group"
        )
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
        _validate_single_process(self.policy)
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad = self.group.all_reduce_sum(parameter.grad)


def _validate_single_process(policy: DistributedPolicy) -> None:
    if policy.world_size != 1:
        raise RuntimeError(
            "this collective requires the future NCCL multi-process backend; "
            "the current implementation is a deterministic world_size=1 runtime"
        )
