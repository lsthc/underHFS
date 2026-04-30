from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
BINARY_MAGIC = b"UHFSBIN1"
_HEADER_SIZE_STRUCT = struct.Struct("<Q")


def save_state_dict(path: str | Path, state: dict[str, Any]) -> None:
    payload = {"format": "underhfs.safe-tensors-lite", "version": FORMAT_VERSION, "state": state}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state_dict(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "underhfs.safe-tensors-lite":
        raise ValueError("not an underhfs state file")
    return payload["state"]


def save_binary_state_dict(path: str | Path, state: dict[str, Any]) -> None:
    header: dict[str, Any] = {
        "format": "underhfs.safe-tensors-binary",
        "version": FORMAT_VERSION,
        "tensors": {},
    }
    chunks: list[bytes] = []
    offset = 0
    for name, value in state.items():
        flat = _flatten_state_value(value)
        data = struct.pack(f"<{len(flat)}f", *flat) if flat else b""
        header["tensors"][name] = {
            "dtype": "fp32",
            "shape": _shape_of(value),
            "offset": offset,
            "nbytes": len(data),
        }
        chunks.append(data)
        offset += len(data)
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    Path(path).write_bytes(BINARY_MAGIC + _HEADER_SIZE_STRUCT.pack(len(encoded_header)) + encoded_header + b"".join(chunks))


def load_binary_state_dict(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    if len(data) < len(BINARY_MAGIC) + _HEADER_SIZE_STRUCT.size or not data.startswith(BINARY_MAGIC):
        raise ValueError("not an underhfs binary state file")
    header_size_offset = len(BINARY_MAGIC)
    header_size = _HEADER_SIZE_STRUCT.unpack_from(data, header_size_offset)[0]
    header_offset = header_size_offset + _HEADER_SIZE_STRUCT.size
    payload_offset = header_offset + header_size
    if payload_offset > len(data):
        raise ValueError("truncated underhfs binary state header")
    header = json.loads(data[header_offset:payload_offset].decode("utf-8"))
    if header.get("format") != "underhfs.safe-tensors-binary":
        raise ValueError("not an underhfs binary state file")
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported binary state version: {header.get('version')}")
    out: dict[str, Any] = {}
    for name, meta in header["tensors"].items():
        if meta.get("dtype") != "fp32":
            raise ValueError(f"unsupported tensor dtype in binary state: {meta.get('dtype')}")
        start = payload_offset + int(meta["offset"])
        end = start + int(meta["nbytes"])
        if end > len(data):
            raise ValueError(f"truncated tensor payload: {name}")
        count = int(meta["nbytes"]) // 4
        flat = list(struct.unpack(f"<{count}f", data[start:end])) if count else []
        out[name] = _unflatten_state_value(flat, list(meta["shape"]))
    return out


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


def _flatten_state_value(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_state_value(item))
        return out
    return [float(value)]


def _unflatten_state_value(flat: list[float], shape: list[int]) -> Any:
    if not shape:
        if len(flat) != 1:
            raise ValueError("scalar tensor payload has invalid size")
        return flat[0]
    expected = 1
    for dim in shape:
        expected *= dim
    if expected != len(flat):
        raise ValueError(f"shape {shape} expects {expected} values, got {len(flat)}")

    cursor = 0

    def build(dims: list[int]) -> Any:
        nonlocal cursor
        if not dims:
            value = flat[cursor]
            cursor += 1
            return value
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(shape)
