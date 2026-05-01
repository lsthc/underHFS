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

from live_world import LiveWorldModel, Player, move, render_viewport, training_samples


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>underHFS liveSee</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: Consolas, monospace; }
    main { display: grid; place-items: center; min-height: 100vh; gap: 12px; }
    #world { display: grid; gap: 3px; }
    .tile { width: 32px; height: 32px; display: grid; place-items: center; font-weight: 700; }
    #hud { color: #b7f7d3; }
  </style>
</head>
<body>
<main>
  <div id="hud">WASD to move</div>
  <div id="world"></div>
</main>
<script>
async function state(dir) {
  const url = dir ? `/api/move?dir=${dir}` : "/api/state";
  const data = await fetch(url).then(r => r.json());
  const world = document.getElementById("world");
  world.style.gridTemplateColumns = `repeat(${data.tiles[0].length}, 32px)`;
  world.innerHTML = "";
  for (const row of data.tiles) {
    for (const tile of row) {
      const el = document.createElement("div");
      el.className = "tile";
      el.textContent = tile.char;
      el.title = `${tile.name} (${tile.x}, ${tile.y})`;
      el.style.background = tile.color;
      el.style.color = tile.name === "player" ? "#111" : "#06110a";
      world.appendChild(el);
    }
  }
  document.getElementById("hud").textContent =
    `${data.prompt} | player=(${data.player.x}, ${data.player.y}) | WASD to move`;
}
window.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();
  if ("wasd".includes(key)) state(key);
});
state();
</script>
</body>
</html>
"""


class LiveWorldSession:
    def __init__(self, prompt: str = "glowing water ruins", *, steps: int = 10) -> None:
        self.prompt = prompt
        self.player = Player()
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
        return render_viewport(self.model, self.prompt, self.player, radius=5)

    def move(self, direction: str) -> dict:
        self.player = move(self.player, direction)
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
    server = ThreadingHTTPServer((host, port), make_handler(session))
    return server


def main() -> int:
    server = create_server(port=8766)
    print("liveSee world running at http://127.0.0.1:8766")
    print("Open it in a browser and move with WASD.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
