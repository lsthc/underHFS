from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Iterator


class ServingProtocol(str, Enum):
    PYTHON = "python"
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    CPP = "cpp"


class StreamSourceKind(str, Enum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    HLS = "hls"
    WEBRTC = "webrtc"
    NETWORK = "network"


@dataclass(frozen=True)
class StreamFrame:
    index: int
    data: bytes
    source: str
    kind: StreamSourceKind

    def to_dict(self) -> dict[str, str | int]:
        return {
            "index": self.index,
            "bytes": len(self.data),
            "source": self.source,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class ProtocolCapability:
    protocol: ServingProtocol
    available: bool
    transport: str
    reason: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "protocol": self.protocol.value,
            "available": self.available,
            "transport": self.transport,
            "reason": self.reason,
        }


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    protocols: tuple[ServingProtocol, ...] = (ServingProtocol.PYTHON, ServingProtocol.HTTP, ServingProtocol.WEBSOCKET)


class PythonServer:
    def __init__(self, handler: Callable[[Any], Any], config: ServeConfig | None = None) -> None:
        self.handler = handler
        self.config = config or ServeConfig()

    def predict(self, payload: Any) -> Any:
        return self.handler(payload)


class JsonHTTPServer:
    def __init__(self, handler: Callable[[Any], Any], config: ServeConfig | None = None) -> None:
        self.handler = handler
        self.config = config or ServeConfig()
        request_handler = self._request_handler()
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), request_handler)
        self.host, self.port = self._server.server_address
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "JsonHTTPServer":
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

    def _request_handler(self):
        predict = self.handler

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/health":
                    self._send_json({"error": "not found"}, status=404)
                    return
                self._send_json({"status": "ok"})

            def do_POST(self) -> None:
                if self.path not in {"/predict", "/v1/predict"}:
                    self._send_json({"error": "not found"}, status=404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    self._send_json({"result": predict(payload)})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)

            def log_message(self, *_args) -> None:
                return

            def _send_json(self, payload: Any, status: int = 200) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


class JsonWebSocketServer(PythonServer):
    def predict_frame(self, text_frame: str) -> str:
        payload = json.loads(text_frame)
        return json.dumps({"result": self.predict(payload)})


@dataclass
class ServingManifest:
    protocol: ServingProtocol
    entrypoint: str
    config: ServeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.value,
            "entrypoint": self.entrypoint,
            "host": self.config.host,
            "port": self.config.port,
        }


def serve(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> PythonServer:
    return PythonServer(handler, config)


def serve_http(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> JsonHTTPServer:
    return JsonHTTPServer(handler, config)


def serve_websocket(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> JsonWebSocketServer:
    return JsonWebSocketServer(handler, config)


def serve_grpc_manifest(config: ServeConfig | None = None) -> ServingManifest:
    return ServingManifest(ServingProtocol.GRPC, "underhfs.grpc.JsonPredictService", config or ServeConfig())


def serve_cpp_manifest(config: ServeConfig | None = None) -> ServingManifest:
    return ServingManifest(ServingProtocol.CPP, "underhfs_cpp_serve", config or ServeConfig())


def protocol_capabilities() -> list[ProtocolCapability]:
    return [
        ProtocolCapability(ServingProtocol.PYTHON, True, "in-process callable"),
        ProtocolCapability(ServingProtocol.HTTP, True, "standard-library JSON HTTP"),
        ProtocolCapability(
            ServingProtocol.WEBSOCKET,
            True,
            "JSON frame adapter",
            "network upgrade loop is planned; frame-level serving is available",
        ),
        ProtocolCapability(
            ServingProtocol.GRPC,
            True,
            "service manifest",
            "emits a stable service manifest until grpcio/protobuf runtime is installed",
        ),
        ProtocolCapability(
            ServingProtocol.CPP,
            True,
            "native executable manifest",
            "emits a stable C++ serving manifest while the executable is built",
        ),
    ]


def require_protocol(protocol: ServingProtocol | str) -> None:
    actual = ServingProtocol(protocol)
    for capability in protocol_capabilities():
        if capability.protocol is actual:
            if capability.available:
                return
            raise RuntimeError(f"{actual.value} serving is unavailable: {capability.reason}")
    raise RuntimeError(f"unknown serving protocol: {actual.value}")


def serve_protocol(
    handler: Callable[[Any], Any],
    protocol: ServingProtocol | str,
    config: ServeConfig | None = None,
) -> PythonServer | JsonHTTPServer:
    actual = ServingProtocol(protocol)
    require_protocol(actual)
    if actual is ServingProtocol.PYTHON:
        return serve(handler, config)
    if actual is ServingProtocol.HTTP:
        return serve_http(handler, config)
    if actual is ServingProtocol.WEBSOCKET:
        return serve_websocket(handler, config)
    raise RuntimeError(f"{actual.value} serving uses manifest generation instead of in-process prediction")


def open_stream(source: str, kind: StreamSourceKind = StreamSourceKind.FILE, *, chunk_bytes: int = 4096) -> Iterator[StreamFrame]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if kind is StreamSourceKind.FILE:
        path = Path(source)
        with path.open("rb") as handle:
            index = 0
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                yield StreamFrame(index=index, data=chunk, source=str(path), kind=kind)
                index += 1
        return
    if kind in {StreamSourceKind.RTSP, StreamSourceKind.HLS, StreamSourceKind.WEBRTC, StreamSourceKind.WEBCAM}:
        raise RuntimeError(f"{kind.value} streaming requires optional FFmpeg/OpenCV/WebRTC integration")
    raise RuntimeError(f"{kind.value} streaming requires a configured network transport")
