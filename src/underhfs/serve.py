from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Callable


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
                if self.path != "/predict":
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


def serve(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> PythonServer:
    return PythonServer(handler, config)


def serve_http(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> JsonHTTPServer:
    return JsonHTTPServer(handler, config)


def open_stream(source: str, kind: StreamSourceKind = StreamSourceKind.FILE):
    raise NotImplementedError(
        f"{kind.value} streaming requires optional FFmpeg/OpenCV runtime integration"
    )
