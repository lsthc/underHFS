from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from underhfs.functional import cross_entropy
from underhfs.optim import SGD
from underhfs.tensor import tensor

from live_world import LiveWorldModel, move, render_chunk, spawn_player, training_samples


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>underHFS liveSee voxel world</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #7ec8ff; }
    body { font-family: Consolas, monospace; color: #fff; }
    canvas { width: 100vw; height: 100vh; display: block; background: linear-gradient(#78c7ff, #d8f6ff 52%, #222 52%); }
    #hud { position: fixed; left: 16px; top: 14px; padding: 10px 12px; background: rgba(0,0,0,.45); border: 1px solid rgba(255,255,255,.25); }
  </style>
</head>
<body>
<canvas id="view" width="960" height="540"></canvas>
<div id="hud">WASD move | Q/E or arrows turn</div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
let state = null;

async function load(dir) {
  const url = dir ? `/api/move?dir=${encodeURIComponent(dir)}` : "/api/state";
  state = await fetch(url).then(r => r.json());
  draw();
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) + amount;
  let g = ((n >> 8) & 255) + amount;
  let b = (n & 255) + amount;
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  return `rgb(${r},${g},${b})`;
}

function iso(x, y, z, p) {
  const dx = x - p.x;
  const dz = z - p.z;
  const c = Math.cos(-p.yaw);
  const s = Math.sin(-p.yaw);
  const rx = dx * c - dz * s;
  const rz = dx * s + dz * c;
  const scale = 26;
  return {
    x: canvas.width / 2 + (rx - rz) * scale,
    y: canvas.height * 0.58 + (rx + rz) * scale * 0.42 - y * scale
  };
}

function cube(block, p) {
  const a = iso(block.x, block.y + 1, block.z, p);
  const b = iso(block.x + 1, block.y + 1, block.z, p);
  const c = iso(block.x + 1, block.y + 1, block.z + 1, p);
  const d = iso(block.x, block.y + 1, block.z + 1, p);
  const e = iso(block.x, block.y, block.z, p);
  const f = iso(block.x + 1, block.y, block.z, p);
  const g = iso(block.x + 1, block.y, block.z + 1, p);
  const h = iso(block.x, block.y, block.z + 1, p);
  poly([a,b,c,d], block.color);
  poly([d,c,g,h], shade(block.color, -32));
  poly([b,c,g,f], shade(block.color, -52));
}

function poly(points, color) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,.2)";
  ctx.stroke();
}

function drawCrosshair() {
  ctx.strokeStyle = "rgba(255,255,255,.8)";
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2 - 8, canvas.height / 2);
  ctx.lineTo(canvas.width / 2 + 8, canvas.height / 2);
  ctx.moveTo(canvas.width / 2, canvas.height / 2 - 8);
  ctx.lineTo(canvas.width / 2, canvas.height / 2 + 8);
  ctx.stroke();
}

function draw() {
  if (!state) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const p = state.player;
  const blocks = [...state.blocks].sort((a, b) => (a.x + a.z + a.y) - (b.x + b.z + b.y));
  for (const block of blocks) cube(block, p);
  drawCrosshair();
  document.getElementById("hud").textContent =
    `${state.prompt} | ${state.mode} | pos=(${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}) yaw=${p.yaw.toFixed(2)} | WASD move, Q/E turn`;
}

window.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();
  const map = {arrowleft: "q", arrowright: "e"};
  const dir = map[key] || key;
  if (["w", "a", "s", "d", "q", "e"].includes(dir)) load(dir);
});

load();
</script>
</body>
</html>
"""


class LiveWorldSession:
    def __init__(self, prompt: str = "glowing water ruins", *, steps: int = 10) -> None:
        self.prompt = prompt
        self.player = spawn_player(prompt)
        self.model = LiveWorldModel()
        features, labels = training_samples(prompt)
        x = tensor(features)
        y = tensor(labels)
        opt = SGD(self.model.parameters(), lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = cross_entropy(self.model(x), y)
            loss.backward()
            opt.step()

    def state(self) -> dict:
        return render_chunk(self.model, self.prompt, self.player, radius=8)

    def move(self, direction: str) -> dict:
        self.player = move(self.player, direction, self.prompt)
        return self.state()


def make_handler(session: LiveWorldSession):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send_json(session.state())
                return
            if parsed.path == "/api/move":
                direction = parse_qs(parsed.query).get("dir", [""])[0]
                self._send_json(session.move(direction))
                return
            self.send_error(404)

        def log_message(self, *_args) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8766, prompt: str = "glowing water ruins"):
    session = LiveWorldSession(prompt)
    return ThreadingHTTPServer((host, port), make_handler(session))


def main() -> int:
    server = create_server(port=8766)
    print("liveSee Minecraft-like voxel world running at http://127.0.0.1:8766")
    print("Open it in a browser. WASD moves; Q/E or arrow keys turn.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
