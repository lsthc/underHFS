from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


def init_process_group(policy: DistributedPolicy | None = None) -> DistributedPolicy:
    return policy or DistributedPolicy()


class DistributedDataParallel:
    def __init__(self, module, policy: DistributedPolicy | None = None) -> None:
        self.module = module
        self.policy = init_process_group(policy)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
