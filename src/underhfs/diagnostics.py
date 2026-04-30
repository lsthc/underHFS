from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from shutil import which
from subprocess import run

from underhfs.cuda import device_count, is_available
from underhfs.native import probe as native_probe
from underhfs.native import status as native_status


@dataclass
class DoctorReport:
    python: str
    platform: str
    cuda_visible: bool
    cuda_device_count: int
    native_core: bool
    native_cuda: bool
    native_probe: dict | None
    tools: dict[str, str | None]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "cuda_visible": self.cuda_visible,
            "cuda_device_count": self.cuda_device_count,
            "native_core": self.native_core,
            "native_cuda": self.native_cuda,
            "native_probe": self.native_probe,
            "tools": self.tools,
            "warnings": self.warnings,
        }


def doctor() -> DoctorReport:
    native = native_status()
    tools = {name: which(name) for name in ("git", "cmake", "ninja", "nvcc", "cl", "nvidia-smi")}
    warnings: list[str] = []
    if not native.available:
        warnings.append(f"native core unavailable: {native.reason}")
    if tools["nvcc"] is None:
        warnings.append("CUDA Toolkit nvcc is not on PATH; CUDA extension builds are disabled.")
    if tools["ninja"] is None:
        warnings.append("Ninja is not on PATH; native builds may fall back or fail.")
    if is_available() and tools["nvcc"] is None:
        warnings.append("NVIDIA driver is visible, but CUDA Toolkit is missing from PATH.")
    probe_result = None
    if native.available:
        if not native.cuda_enabled:
            warnings.append("native core is installed, but CUDA support is disabled in this build.")
        try:
            probe_result = native_probe()
        except Exception as exc:
            warnings.append(f"native probe failed: {exc}")
    return DoctorReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cuda_visible=is_available(),
        cuda_device_count=device_count(),
        native_core=native.available,
        native_cuda=native.cuda_enabled,
        native_probe=probe_result,
        tools=tools,
        warnings=warnings,
    )


def nvidia_smi_summary() -> str | None:
    if which("nvidia-smi") is None:
        return None
    result = run(["nvidia-smi"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()
