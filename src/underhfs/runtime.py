from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from underhfs.cuda import MemoryPolicy, MemoryTier, memory_budgets
from underhfs.tensor import DType, Tensor, tensor


_DTYPE_BYTES = {
    DType.FP32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP8_E4M3: 1,
    DType.FP8_E5M2: 1,
    DType.INT8: 1,
    DType.INT4: 0.5,
}


@dataclass(frozen=True)
class Placement:
    tier: MemoryTier
    bytes: int
    reason: str


@dataclass(frozen=True)
class OffloadHandle:
    id: str
    tier: MemoryTier
    path: str
    bytes: int
    shape: tuple[int, ...]
    dtype: DType
    device: str
    sha256: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, str | int | list[int]]:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "path": self.path,
            "bytes": self.bytes,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "device": self.device,
            "sha256": self.sha256,
            "cached": self.cached,
        }


@dataclass
class TierBudget:
    tier: MemoryTier
    capacity_bytes: int
    used_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes)

    def reserve(self, size_bytes: int) -> bool:
        if size_bytes > self.available_bytes:
            return False
        self.used_bytes += size_bytes
        return True


class MemoryPlanner:
    def __init__(self, policy: MemoryPolicy, budgets: dict[MemoryTier, int]) -> None:
        self.policy = policy
        self.budgets = {
            tier: TierBudget(tier, budgets.get(tier, 0))
            for tier in policy.tiers
        }

    def tensor_size_bytes(self, tensor: Tensor) -> int:
        return int(tensor.numel() * _DTYPE_BYTES[tensor.dtype])

    def place_tensor(self, tensor: Tensor) -> Placement:
        return self.place_bytes(self.tensor_size_bytes(tensor))

    def place_bytes(self, size: int) -> Placement:
        if size < 0:
            raise ValueError("size must be non-negative")
        for tier in self.policy.tiers:
            budget = self.budgets[tier]
            if budget.reserve(size):
                return Placement(tier=tier, bytes=size, reason="fits-budget")
        if not self.policy.allow_offload:
            raise MemoryError(f"tensor requires {size} bytes, but no configured tier has capacity")
        final_tier = self.policy.tiers[-1]
        self.budgets[final_tier].used_bytes += size
        return Placement(tier=final_tier, bytes=size, reason="oversubscribed-offload")

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            tier.value: {
                "capacity_bytes": budget.capacity_bytes,
                "used_bytes": budget.used_bytes,
                "available_bytes": budget.available_bytes,
            }
            for tier, budget in self.budgets.items()
        }


class OffloadExecutor:
    def __init__(self, policy: MemoryPolicy, root: str | Path | None = None) -> None:
        self.policy = policy
        scratch = root or policy.scratch_path or ".underhfs-offload"
        self.root = Path(scratch)
        self.root.mkdir(parents=True, exist_ok=True)
        self._prefetch_cache: dict[str, Tensor] = {}

    def offload_tensor(self, value: Tensor, tier: MemoryTier = MemoryTier.NVME) -> OffloadHandle:
        if tier not in {MemoryTier.NVME, MemoryTier.NETWORK}:
            raise ValueError("offload executor currently supports NVMe/network tiers")
        if tier is MemoryTier.NETWORK:
            raise RuntimeError("network offload requires a configured remote transport")
        ident = uuid4().hex
        path = self.root / f"{ident}.uhfsoffload.json"
        data = value.detach().tolist()
        data_bytes = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = {
            "format": "underhfs.offload-tensor",
            "version": 1,
            "id": ident,
            "shape": list(value.shape),
            "dtype": value.dtype.value,
            "device": str(value.device),
            "data_sha256": sha256(data_bytes).hexdigest(),
            "data": data,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        path.write_bytes(encoded)
        return OffloadHandle(
            id=ident,
            tier=tier,
            path=str(path),
            bytes=len(encoded),
            shape=value.shape,
            dtype=value.dtype,
            device=str(value.device),
            sha256=sha256(encoded).hexdigest(),
        )

    def prefetch_tensor(self, handle: OffloadHandle) -> OffloadHandle:
        self._prefetch_cache[handle.id] = self.load_tensor(handle, use_cache=False)
        return OffloadHandle(
            id=handle.id,
            tier=handle.tier,
            path=handle.path,
            bytes=handle.bytes,
            shape=handle.shape,
            dtype=handle.dtype,
            device=handle.device,
            sha256=handle.sha256,
            cached=True,
        )

    def load_tensor(
        self,
        handle: OffloadHandle,
        *,
        device: str | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        if use_cache and handle.id in self._prefetch_cache:
            cached = self._prefetch_cache[handle.id].detach()
            return cached.to(device or "cpu")
        raw = Path(handle.path).read_bytes()
        if handle.sha256 and sha256(raw).hexdigest() != handle.sha256:
            raise ValueError("offload payload checksum mismatch")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("format") != "underhfs.offload-tensor":
            raise ValueError("not an underhfs offload tensor")
        if payload.get("version") != 1:
            raise ValueError(f"unsupported offload tensor version: {payload.get('version')}")
        if payload.get("id") != handle.id:
            raise ValueError("offload handle id mismatch")
        data_bytes = json.dumps(payload["data"], separators=(",", ":"), sort_keys=True).encode("utf-8")
        if sha256(data_bytes).hexdigest() != payload.get("data_sha256"):
            raise ValueError("offload tensor data checksum mismatch")
        if tuple(payload["shape"]) != handle.shape:
            raise ValueError("offload tensor shape mismatch")
        out = tensor(payload["data"], dtype=handle.dtype)
        return out.to(device) if device is not None else out

    def release(self, handle: OffloadHandle) -> None:
        self._prefetch_cache.pop(handle.id, None)
        Path(handle.path).unlink(missing_ok=True)

    def cache_info(self) -> dict[str, int]:
        return {"prefetched_tensors": len(self._prefetch_cache)}


class NetworkOffloadServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        handler = self._handler()
        self._server = ThreadingHTTPServer((host, port), handler)
        self.host, self.port = self._server.server_address
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "NetworkOffloadServer":
        if self._thread is None:
            self._thread = Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _handler(self):
        store = self._store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send_json({"status": "ok", "tensors": len(store)})
                    return
                if self.path.startswith("/load/"):
                    ident = self.path.rsplit("/", 1)[-1]
                    if ident not in store:
                        self._send_json({"error": "not found"}, status=404)
                        return
                    self._send_json(store[ident])
                    return
                self._send_json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                if self.path != "/upload":
                    self._send_json({"error": "not found"}, status=404)
                    return
                try:
                    payload = self._read_json()
                    _validate_network_payload(payload)
                    store[payload["id"]] = payload
                    self._send_json({"ok": True, "id": payload["id"]})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)

            def do_DELETE(self) -> None:
                if not self.path.startswith("/release/"):
                    self._send_json({"error": "not found"}, status=404)
                    return
                ident = self.path.rsplit("/", 1)[-1]
                store.pop(ident, None)
                self._send_json({"ok": True, "id": ident})

            def log_message(self, *_args) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

            def _send_json(self, payload: Any, status: int = 200) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


class NetworkOffloadClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def offload_tensor(self, value: Tensor) -> OffloadHandle:
        ident = uuid4().hex
        data = value.detach().tolist()
        data_bytes = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = {
            "format": "underhfs.network-offload-tensor",
            "version": 1,
            "id": ident,
            "shape": list(value.shape),
            "dtype": value.dtype.value,
            "device": str(value.device),
            "data_sha256": sha256(data_bytes).hexdigest(),
            "data": data,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._request("/upload", method="POST", data=encoded)
        return OffloadHandle(
            id=ident,
            tier=MemoryTier.NETWORK,
            path=f"{self.base_url}/load/{ident}",
            bytes=len(encoded),
            shape=value.shape,
            dtype=value.dtype,
            device=str(value.device),
            sha256=sha256(encoded).hexdigest(),
        )

    def load_tensor(self, handle: OffloadHandle, *, device: str | None = None) -> Tensor:
        payload = self._request(f"/load/{handle.id}")
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if handle.sha256 and sha256(encoded).hexdigest() != handle.sha256:
            raise ValueError("network offload payload checksum mismatch")
        _validate_network_payload(payload)
        if tuple(payload["shape"]) != handle.shape:
            raise ValueError("network offload tensor shape mismatch")
        out = tensor(payload["data"], dtype=handle.dtype)
        return out.to(device) if device is not None else out

    def release(self, handle: OffloadHandle) -> None:
        self._request(f"/release/{handle.id}", method="DELETE")

    def _request(self, path: str, *, method: str = "GET", data: bytes | None = None) -> Any:
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(str(payload["error"])) from exc
            raise RuntimeError(f"HTTP {exc.code}") from exc
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload


def _validate_network_payload(payload: dict[str, Any]) -> None:
    if payload.get("format") != "underhfs.network-offload-tensor":
        raise ValueError("not an underhfs network offload tensor")
    if payload.get("version") != 1:
        raise ValueError(f"unsupported network offload tensor version: {payload.get('version')}")
    data_bytes = json.dumps(payload["data"], separators=(",", ":"), sort_keys=True).encode("utf-8")
    if sha256(data_bytes).hexdigest() != payload.get("data_sha256"):
        raise ValueError("network offload tensor data checksum mismatch")


def planner_from_system(policy: MemoryPolicy | None = None, *, vram_fraction: float = 0.9) -> MemoryPlanner:
    actual_policy = policy or MemoryPolicy()
    return MemoryPlanner(actual_policy, budgets=memory_budgets(vram_fraction=vram_fraction))
