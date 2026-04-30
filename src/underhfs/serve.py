from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


def serve(handler: Callable[[Any], Any], config: ServeConfig | None = None) -> PythonServer:
    return PythonServer(handler, config)


def open_stream(source: str, kind: StreamSourceKind = StreamSourceKind.FILE):
    raise NotImplementedError(
        f"{kind.value} streaming requires optional FFmpeg/OpenCV runtime integration"
    )
