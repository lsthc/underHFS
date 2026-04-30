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


def save_checkpoint(
    path: str | Path,
    *,
    state: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": "underhfs.checkpoint",
        "version": FORMAT_VERSION,
        "metadata": metadata or {},
        "state": state,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "underhfs.checkpoint":
        raise ValueError("not an underhfs checkpoint")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    return payload


def export_onnx(*args, **kwargs) -> None:
    raise NotImplementedError("ONNX export is planned for the native graph IR backend")


def export_manifest(
    path: str | Path,
    *,
    model_name: str,
    state: dict[str, Any],
    inputs: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": "underhfs.export-manifest",
        "version": FORMAT_VERSION,
        "model": model_name,
        "inputs": inputs or {},
        "parameters": [
            {"name": name, "shape": _shape_of(value)}
            for name, value in state.items()
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "underhfs.export-manifest":
        raise ValueError("not an underhfs export manifest")
    return payload


def _shape_of(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value), *_shape_of(value[0])]
    return []
