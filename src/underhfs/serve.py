from __future__ import annotations

import json
import base64
from hashlib import sha1
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
from shutil import which
from subprocess import PIPE, Popen
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


class WebSocketPredictServer(JsonWebSocketServer):
    def __init__(self, handler: Callable[[Any], Any], config: ServeConfig | None = None) -> None:
        super().__init__(handler, config)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.host, self.config.port))
        self._socket.listen(16)
        self.host, self.port = self._socket.getsockname()
        self._thread: Thread | None = None
        self._closed = False

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> "WebSocketPredictServer":
        if self._thread is None:
            self._thread = Thread(target=self._serve_forever, daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _serve_forever(self) -> None:
        while not self._closed:
            try:
                client, _ = self._socket.accept()
            except OSError:
                break
            Thread(target=self._handle_client, args=(client,), daemon=True).start()

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            request = client.recv(4096).decode("utf-8", errors="ignore")
            key = _websocket_header(request, "Sec-WebSocket-Key")
            if not key:
                return
            accept = base64.b64encode(sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()).decode("ascii")
            client.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode("ascii")
            )
            while not self._closed:
                text = _recv_ws_text(client)
                if text is None:
                    break
                client.sendall(_encode_ws_text(self.predict_frame(text)))


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


def serve_websocket_loop(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> WebSocketPredictServer:
    return WebSocketPredictServer(handler, config)


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
            "standard-library WebSocket JSON server",
            "supports text-frame predict loops for local serving",
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
    if kind is StreamSourceKind.WEBCAM:
        yield from _open_opencv_stream(source, kind, chunk_bytes=chunk_bytes)
        return
    if kind in {StreamSourceKind.RTSP, StreamSourceKind.HLS}:
        yield from _open_ffmpeg_stream(source, kind, chunk_bytes=chunk_bytes)
        return
    if kind is StreamSourceKind.WEBRTC:
        raise RuntimeError("webrtc streaming requires an optional WebRTC transport adapter")
    raise RuntimeError(f"{kind.value} streaming requires a configured network transport")


_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _websocket_header(request: str, name: str) -> str:
    prefix = f"{name.lower()}:"
    for line in request.splitlines():
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_ws_text(sock: socket.socket) -> str | None:
    header = _recv_exact(sock, 2)
    if header is None:
        return None
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F
    if opcode == 0x8:
        return None
    if opcode != 0x1:
        raise RuntimeError("only WebSocket text frames are supported")
    if length == 126:
        extended = _recv_exact(sock, 2)
        if extended is None:
            return None
        length = int.from_bytes(extended, "big")
    elif length == 127:
        extended = _recv_exact(sock, 8)
        if extended is None:
            return None
        length = int.from_bytes(extended, "big")
    mask = _recv_exact(sock, 4) if masked else b"\x00\x00\x00\x00"
    payload = _recv_exact(sock, length)
    if payload is None or mask is None:
        return None
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return payload.decode("utf-8")


def _encode_ws_text(text: str) -> bytes:
    payload = text.encode("utf-8")
    if len(payload) < 126:
        return bytes([0x81, len(payload)]) + payload
    if len(payload) < 65536:
        return bytes([0x81, 126]) + len(payload).to_bytes(2, "big") + payload
    return bytes([0x81, 127]) + len(payload).to_bytes(8, "big") + payload


def _open_ffmpeg_stream(source: str, kind: StreamSourceKind, *, chunk_bytes: int) -> Iterator[StreamFrame]:
    if which("ffmpeg") is None:
        raise RuntimeError(f"{kind.value} streaming requires ffmpeg on PATH")
    process = Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", source, "-f", "rawvideo", "-"],
        stdout=PIPE,
        stderr=PIPE,
    )
    assert process.stdout is not None
    try:
        index = 0
        while True:
            chunk = process.stdout.read(chunk_bytes)
            if not chunk:
                break
            yield StreamFrame(index=index, data=chunk, source=source, kind=kind)
            index += 1
    finally:
        process.terminate()


def _open_opencv_stream(source: str, kind: StreamSourceKind, *, chunk_bytes: int) -> Iterator[StreamFrame]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("webcam streaming requires opencv-python") from exc
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open OpenCV stream source: {source}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            data = frame.tobytes()
            for offset in range(0, len(data), chunk_bytes):
                yield StreamFrame(index=index, data=data[offset : offset + chunk_bytes], source=source, kind=kind)
                index += 1
    finally:
        capture.release()
