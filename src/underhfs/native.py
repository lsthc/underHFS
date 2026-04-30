from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    reason: str | None = None
    cuda_enabled: bool = False


def status() -> NativeStatus:
    try:
        _prepare_windows_cuda_dll_search_path()
        import underhfs._core as _core
    except Exception as exc:
        return NativeStatus(False, str(exc))
    return NativeStatus(True, None, bool(getattr(_core, "cuda_enabled", False)))


def require_native():
    state = status()
    if not state.available:
        raise RuntimeError(f"underHFS native core is unavailable: {state.reason}")
    _prepare_windows_cuda_dll_search_path()
    import underhfs._core as _core

    return _core


def _prepare_windows_cuda_dll_search_path() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
        candidates.append(Path(cuda_path) / "bin" / "x64")
    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_root.exists():
        for path in sorted(cuda_root.glob("v*"), reverse=True):
            candidates.append(path / "bin")
            candidates.append(path / "bin" / "x64")
    for candidate in candidates:
        path = str(candidate)
        if candidate.exists() and path not in _DLL_DIRECTORY_PATHS:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path))
            _DLL_DIRECTORY_PATHS.add(path)


def probe() -> dict[str, Any]:
    core = require_native()
    left = core.TensorCore([1.0, 2.0, 3.0, 4.0], [2, 2])
    right = core.TensorCore([5.0, 6.0, 7.0, 8.0], [2, 2])
    added = left.add(right)
    product = left.matmul(right)
    cuda_enabled = bool(getattr(core, "cuda_enabled", False))
    result = {
        "version": getattr(core, "__version__", "unknown"),
        "cuda_enabled": cuda_enabled,
        "numel": left.numel(),
        "shape": list(left.shape),
        "strides": list(left.strides),
        "add": list(added.storage),
        "matmul": list(product.storage),
        "sum": list(product.sum().storage),
    }
    if cuda_enabled and hasattr(core, "cuda_add_f32"):
        result["cuda_add_f32"] = list(core.cuda_add_f32([1.0, 2.0], [3.0, 4.0]))
    if cuda_enabled and hasattr(core, "CudaTensorF32"):
        left_gpu = core.CudaTensorF32([1.0, 2.0], [2])
        right_gpu = core.CudaTensorF32([3.0, 4.0], [2])
        result["cuda_tensor_add_f32"] = list(left_gpu.add(right_gpu).to_host())
        result["cuda_tensor_mul_f32"] = list(left_gpu.mul(right_gpu).to_host())
        result["cuda_tensor_sum_f32"] = list(left_gpu.sum().to_host())
        left_gpu_matmul = core.CudaTensorF32([1.0, 2.0, 3.0, 4.0], [2, 2])
        right_gpu_matmul = core.CudaTensorF32([5.0, 6.0, 7.0, 8.0], [2, 2])
        result["cuda_tensor_matmul_f32"] = list(left_gpu_matmul.matmul(right_gpu_matmul).to_host())
        if hasattr(core, "cuda_fused_adamw_f32"):
            result["cuda_fused_adamw_f32"] = {
                key: list(value)
                for key, value in core.cuda_fused_adamw_f32(
                    [1.0, 2.0],
                    [0.1, 0.2],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    0.01,
                    0.9,
                    0.999,
                    1e-8,
                    0.0,
                    1,
                ).items()
            }
        if hasattr(core, "cuda_attention_f32"):
            result["cuda_attention_f32"] = list(
                core.cuda_attention_f32(
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 2.0, 3.0, 4.0],
                    2,
                    2,
                    1.0,
                    False,
                )
            )
        if hasattr(core, "CudaTensorF16"):
            left_f16 = core.CudaTensorF16([1.0, 2.0], [2])
            right_f16 = core.CudaTensorF16([3.0, 4.0], [2])
            result["cuda_tensor_add_f16"] = list(left_f16.add(right_f16).to_host())
            result["cuda_tensor_mul_f16"] = list(left_f16.mul(right_f16).to_host())
        if hasattr(core, "CudaTensorBF16"):
            left_bf16 = core.CudaTensorBF16([1.0, 2.0], [2])
            right_bf16 = core.CudaTensorBF16([3.0, 4.0], [2])
            result["cuda_tensor_add_bf16"] = list(left_bf16.add(right_bf16).to_host())
            result["cuda_tensor_mul_bf16"] = list(left_bf16.mul(right_bf16).to_host())
        if hasattr(core, "cuda_allocator_stats"):
            result["cuda_allocator"] = {
                str(key): int(value) for key, value in core.cuda_allocator_stats().items()
            }
        if hasattr(core, "cuda_stream_stats"):
            result["cuda_stream"] = {
                str(key): int(value) for key, value in core.cuda_stream_stats().items()
            }
    return result
