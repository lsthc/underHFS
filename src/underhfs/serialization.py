from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1


def save_state_dict(path: str | Path, state: dict[str, Any]) -> None:
    payload = {"format": "underhfs.safe-tensors-lite", "version": FORMAT_VERSION, "state": state}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state_dict(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "underhfs.safe-tensors-lite":
        raise ValueError("not an underhfs state file")
    return payload["state"]


def export_onnx(*args, **kwargs) -> None:
    raise NotImplementedError("ONNX export is planned for the native graph IR backend")
