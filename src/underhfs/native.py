from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    reason: str | None = None
    cuda_enabled: bool = False


def status() -> NativeStatus:
    try:
        import underhfs._core as _core
    except Exception as exc:
        return NativeStatus(False, str(exc))
    return NativeStatus(True, None, bool(getattr(_core, "cuda_enabled", False)))


def require_native():
    state = status()
    if not state.available:
        raise RuntimeError(f"underHFS native core is unavailable: {state.reason}")
    import underhfs._core as _core

    return _core


def probe() -> dict[str, Any]:
    core = require_native()
    left = core.TensorCore([1.0, 2.0, 3.0, 4.0], [2, 2])
    right = core.TensorCore([5.0, 6.0, 7.0, 8.0], [2, 2])
    added = left.add(right)
    product = left.matmul(right)
    return {
        "version": getattr(core, "__version__", "unknown"),
        "cuda_enabled": bool(getattr(core, "cuda_enabled", False)),
        "numel": left.numel(),
        "shape": list(left.shape),
        "strides": list(left.strides),
        "add": list(added.storage),
        "matmul": list(product.storage),
        "sum": list(product.sum().storage),
    }
