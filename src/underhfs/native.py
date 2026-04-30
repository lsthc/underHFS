from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    reason: str | None = None


def status() -> NativeStatus:
    try:
        import underhfs._core as _core  # noqa: F401
    except Exception as exc:
        return NativeStatus(False, str(exc))
    return NativeStatus(True, None)


def require_native():
    state = status()
    if not state.available:
        raise RuntimeError(f"underHFS native core is unavailable: {state.reason}")
    import underhfs._core as _core

    return _core
