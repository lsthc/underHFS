from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from shutil import which
from subprocess import run


class MemoryTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"
    NETWORK = "network"


@dataclass
class MemoryPolicy:
    tiers: tuple[MemoryTier, ...] = (MemoryTier.VRAM, MemoryTier.RAM)
    allow_offload: bool = True
    allow_prefetch: bool = True
    activation_checkpointing: bool = True
    recompute: bool = True
    scratch_path: str | None = None


@dataclass
class PrecisionPolicy:
    strict: bool = False
    allow_fp8: bool = True
    allow_int4: bool = True
    loss_scaling: str = "dynamic"


@dataclass
class RuntimePolicy:
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    stream_aware: bool = True


def is_available() -> bool:
    return which("nvidia-smi") is not None


def device_count() -> int:
    if not is_available():
        return 0
    result = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return 0 if result.returncode else len([line for line in result.stdout.splitlines() if line.strip()])


def require_cuda_toolkit() -> None:
    if which("nvcc") is None:
        raise RuntimeError(
            "CUDA Toolkit nvcc was not found on PATH. Install CUDA Toolkit 13.x and open a developer shell."
        )
