from __future__ import annotations

import json
from hashlib import sha256
import struct
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
BINARY_MAGIC = b"UHFSBIN1"
_HEADER_SIZE_STRUCT = struct.Struct("<Q")
_BINARY_FORMAT = "underhfs.safe-tensors-binary"


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
        "format": _BINARY_FORMAT,
        "version": FORMAT_VERSION,
        "payload_nbytes": 0,
        "payload_sha256": "",
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
            "sha256": sha256(data).hexdigest(),
        }
        chunks.append(data)
        offset += len(data)
    payload = b"".join(chunks)
    header["payload_nbytes"] = len(payload)
    header["payload_sha256"] = sha256(payload).hexdigest()
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    Path(path).write_bytes(BINARY_MAGIC + _HEADER_SIZE_STRUCT.pack(len(encoded_header)) + encoded_header + payload)


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
    if header.get("format") != _BINARY_FORMAT:
        raise ValueError("not an underhfs binary state file")
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported binary state version: {header.get('version')}")
    payload = data[payload_offset:]
    expected_payload_nbytes = int(header.get("payload_nbytes", len(payload)))
    if expected_payload_nbytes != len(payload):
        raise ValueError(
            f"binary state payload size mismatch: expected {expected_payload_nbytes} bytes, got {len(payload)}"
        )
    expected_payload_hash = header.get("payload_sha256")
    if expected_payload_hash and sha256(payload).hexdigest() != expected_payload_hash:
        raise ValueError("binary state payload checksum mismatch")
    _validate_tensor_layout(header["tensors"], len(payload))
    out: dict[str, Any] = {}
    for name, meta in header["tensors"].items():
        if meta.get("dtype") != "fp32":
            raise ValueError(f"unsupported tensor dtype in binary state: {meta.get('dtype')}")
        start = int(meta["offset"])
        end = start + int(meta["nbytes"])
        if end > len(payload):
            raise ValueError(f"truncated tensor payload: {name}")
        tensor_payload = payload[start:end]
        expected_tensor_hash = meta.get("sha256")
        if expected_tensor_hash and sha256(tensor_payload).hexdigest() != expected_tensor_hash:
            raise ValueError(f"tensor payload checksum mismatch: {name}")
        count = int(meta["nbytes"]) // 4
        flat = list(struct.unpack(f"<{count}f", tensor_payload)) if count else []
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


def export_onnx(
    path: str | Path,
    *,
    model_name: str,
    state: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    include_state: bool = True,
) -> None:
    if _try_export_real_onnx(path, model_name=model_name, state=state, inputs=inputs):
        return
    state_json = json.dumps(state, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = {
        "ir_version": 10,
        "producer_name": "underhfs",
        "format": "underhfs.onnx-lite",
        "version": FORMAT_VERSION,
        "state_sha256": sha256(state_json).hexdigest(),
        "graph": {
            "name": model_name,
            "inputs": inputs or {},
            "initializers": [
                {"name": name, "dims": _shape_of(value), "data_type": "FLOAT"}
                for name, value in state.items()
            ],
        },
    }
    if include_state:
        payload["state"] = state
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def import_onnx(path: str | Path) -> dict[str, Any]:
    try:
        return _try_import_real_onnx(path)
    except ImportError:
        pass
    except Exception:
        pass
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "underhfs.onnx-lite":
        raise ValueError("only underhfs.onnx-lite manifests are supported without the optional ONNX runtime")
    if "state" in payload:
        state_json = json.dumps(payload["state"], separators=(",", ":"), sort_keys=True).encode("utf-8")
        if sha256(state_json).hexdigest() != payload.get("state_sha256"):
            raise ValueError("ONNX-lite state checksum mismatch")
    return payload


def load_onnx_state_dict(path: str | Path) -> dict[str, Any]:
    payload = import_onnx(path)
    if "state" not in payload:
        raise ValueError("ONNX-lite manifest does not include embedded state")
    return payload["state"]


def _try_export_real_onnx(
    path: str | Path,
    *,
    model_name: str,
    state: dict[str, Any],
    inputs: dict[str, Any] | None,
) -> bool:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return False
    initializers = []
    for name, value in state.items():
        flat = _flatten_state_value(value)
        initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.FLOAT,
                dims=_shape_of(value),
                vals=flat,
            )
        )
    graph_inputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.FLOAT,
            spec.get("shape", []),
        )
        for name, spec in (inputs or {}).items()
    ]
    graph = helper.make_graph(
        nodes=[],
        name=model_name,
        inputs=graph_inputs,
        outputs=[],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="underhfs")
    model.metadata_props.append(onnx.StringStringEntryProto(key="format", value="underhfs.onnx"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    return True


def _try_import_real_onnx(path: str | Path) -> dict[str, Any]:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise
    model = onnx.load(str(path))
    return {
        "format": "underhfs.onnx",
        "producer_name": model.producer_name,
        "graph": {
            "name": model.graph.name,
            "initializers": [
                {
                    "name": tensor_proto.name,
                    "dims": list(tensor_proto.dims),
                    "data_type": str(tensor_proto.data_type),
                }
                for tensor_proto in model.graph.initializer
            ],
        },
        "state": {
            tensor_proto.name: numpy_helper.to_array(tensor_proto).tolist()
            for tensor_proto in model.graph.initializer
        },
    }


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


def _validate_tensor_layout(tensors: dict[str, Any], payload_nbytes: int) -> None:
    spans: list[tuple[int, int, str]] = []
    for name, meta in tensors.items():
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        if offset < 0 or nbytes < 0:
            raise ValueError(f"invalid negative tensor span: {name}")
        if nbytes % 4 != 0:
            raise ValueError(f"fp32 tensor payload is not 4-byte aligned: {name}")
        end = offset + nbytes
        if end > payload_nbytes:
            raise ValueError(f"tensor payload points past end of file: {name}")
        spans.append((offset, end, name))
    spans.sort()
    previous_end = 0
    previous_name = ""
    for start, end, name in spans:
        if start < previous_end:
            raise ValueError(f"overlapping tensor payloads: {previous_name} and {name}")
        previous_end = end
        previous_name = name
