from __future__ import annotations

import ctypes
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


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    memory_total_bytes: int
    memory_free_bytes: int
    driver_version: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_free_bytes": self.memory_free_bytes,
            "driver_version": self.driver_version,
        }


def is_available() -> bool:
    return which("nvidia-smi") is not None


def device_count() -> int:
    if not is_available():
        return 0
    result = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return 0 if result.returncode else len([line for line in result.stdout.splitlines() if line.strip()])


def devices() -> list[CudaDeviceInfo]:
    if not is_available():
        return []
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return _parse_nvidia_smi_devices(result.stdout)


def memory_budgets(vram_fraction: float = 0.9) -> dict[MemoryTier, int]:
    if not 0.0 < vram_fraction <= 1.0:
        raise ValueError("vram_fraction must be in the range (0, 1]")
    gpu_devices = devices()
    budgets: dict[MemoryTier, int] = {}
    if gpu_devices:
        budgets[MemoryTier.VRAM] = int(sum(device.memory_free_bytes for device in gpu_devices) * vram_fraction)
    ram_available = _available_ram_bytes()
    if ram_available is not None:
        budgets[MemoryTier.RAM] = int(ram_available * 0.8)
    return budgets


def require_cuda_toolkit() -> None:
    if which("nvcc") is None:
        raise RuntimeError(
            "CUDA Toolkit nvcc was not found on PATH. Install CUDA Toolkit 13.x and open a developer shell."
        )


def _parse_nvidia_smi_devices(output: str) -> list[CudaDeviceInfo]:
    devices_out: list[CudaDeviceInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total_mib, free_mib, driver = parts
        try:
            devices_out.append(
                CudaDeviceInfo(
                    index=int(index),
                    name=name,
                    memory_total_bytes=int(float(total_mib) * 1024 * 1024),
                    memory_free_bytes=int(float(free_mib) * 1024 * 1024),
                    driver_version=driver,
                )
            )
        except ValueError:
            continue
    return devices_out


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _available_ram_bytes() -> int | None:
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        return None
