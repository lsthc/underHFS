from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
from shutil import which
from subprocess import run

from underhfs.tensor import DType


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
class KernelCapability:
    op: str
    device: str
    dtypes: tuple[DType, ...]
    forward: str
    backward: str

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "op": self.op,
            "device": self.device,
            "dtypes": [dtype.value for dtype in self.dtypes],
            "forward": self.forward,
            "backward": self.backward,
        }


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


def capability_matrix() -> list[KernelCapability]:
    """Return the runtime contract underHFS currently exposes for CUDA work.

    This is intentionally conservative: unsupported dtype/op combinations are
    reported as missing instead of silently pretending a slow fallback exists.
    """

    from underhfs.native import status

    native = status()
    cpu_dtypes = (
        DType.FP32,
        DType.FP16,
        DType.BF16,
        DType.FP8_E4M3,
        DType.FP8_E5M2,
        DType.INT8,
        DType.INT4,
    )
    capabilities = [
        KernelCapability("add", "cpu", cpu_dtypes, "python/native-fp32", "python-autograd"),
        KernelCapability("mul", "cpu", cpu_dtypes, "python/native-fp32", "python-autograd"),
        KernelCapability("matmul", "cpu", (DType.FP32,), "python/native-fp32", "python-autograd"),
        KernelCapability("sum", "cpu", cpu_dtypes, "python/native-fp32", "python-autograd"),
    ]
    if native.cuda_enabled:
        capabilities.extend(
            [
                KernelCapability(
                    "add",
                    "cuda",
                    (DType.FP32, DType.FP16, DType.BF16),
                    "native-cuda",
                    "python-autograd-preserves-cuda",
                ),
                KernelCapability(
                    "mul",
                    "cuda",
                    (DType.FP32, DType.FP16, DType.BF16),
                    "native-cuda",
                    "python-autograd-preserves-cuda",
                ),
                KernelCapability(
                    "matmul",
                    "cuda",
                    (DType.FP32,),
                    "native-cuda-cublas",
                    "python-autograd-preserves-cuda",
                ),
                KernelCapability(
                    "sum",
                    "cuda",
                    (DType.FP32,),
                    "native-cuda",
                    "python-autograd-preserves-cuda",
                ),
                KernelCapability(
                    "conv2d",
                    "cuda",
                    (DType.FP32,),
                    "native-cudnn" if native.cudnn_enabled else "requires-cudnn",
                    "python-autograd",
                ),
                KernelCapability(
                    "attention",
                    "cuda",
                    (DType.FP32,),
                    "native-cuda-attention",
                    "python-autograd",
                ),
                KernelCapability(
                    "fused_adamw",
                    "cuda",
                    (DType.FP32, DType.FP16, DType.BF16),
                    "native-cuda-fp32/python-other-dtypes",
                    "optimizer-state-preserving",
                ),
            ]
        )
    return capabilities


def supports_kernel(op: str, *, device: str = "cpu", dtype: DType | str = DType.FP32) -> bool:
    actual_dtype = DType(dtype)
    return any(
        capability.op == op and capability.device == device and actual_dtype in capability.dtypes
        for capability in capability_matrix()
    )


def require_kernel(op: str, *, device: str = "cpu", dtype: DType | str = DType.FP32) -> None:
    if supports_kernel(op, device=device, dtype=dtype):
        return
    supported = [
        capability.to_dict()
        for capability in capability_matrix()
        if capability.op == op and capability.device == device
    ]
    raise RuntimeError(
        f"underHFS does not have a {device} {DType(dtype).value} kernel for {op}. "
        f"Supported variants: {supported or 'none'}"
    )


def allocator_stats() -> dict[str, int]:
    from underhfs.native import require_native

    core = require_native()
    if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_allocator_stats"):
        raise RuntimeError("CUDA allocator stats are unavailable in this underHFS native build")
    return {str(key): int(value) for key, value in core.cuda_allocator_stats().items()}


def empty_cache() -> None:
    from underhfs.native import require_native

    core = require_native()
    if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_empty_cache"):
        raise RuntimeError("CUDA empty_cache is unavailable in this underHFS native build")
    core.cuda_empty_cache()


def stream_stats() -> dict[str, int]:
    from underhfs.native import require_native

    core = require_native()
    if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_stream_stats"):
        raise RuntimeError("CUDA stream stats are unavailable in this underHFS native build")
    return {str(key): int(value) for key, value in core.cuda_stream_stats().items()}


def synchronize() -> None:
    from underhfs.native import require_native

    core = require_native()
    if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_synchronize"):
        raise RuntimeError("CUDA synchronize is unavailable in this underHFS native build")
    core.cuda_synchronize()


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
