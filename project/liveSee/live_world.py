from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Iterable

from underhfs.nn import Linear, ReLU, Sequential
from underhfs.tensor import Tensor, tensor


TILES = {
    0: {"char": ".", "name": "grass", "color": "#5abf75"},
    1: {"char": "~", "name": "water", "color": "#4aa3df"},
    2: {"char": "^", "name": "ridge", "color": "#9b8a68"},
    3: {"char": "*", "name": "glow", "color": "#f2c94c"},
    4: {"char": "#", "name": "ruin", "color": "#7f8c8d"},
}


@dataclass(frozen=True)
class Player:
    x: int = 0
    y: int = 0


class LiveWorldModel:
    def __init__(self) -> None:
        self.net = Sequential(Linear(8, 16), ReLU(), Linear(16, len(TILES)))

    def parameters(self):
        return self.net.parameters()

    def __call__(self, features: Tensor) -> Tensor:
        return self.net(features)

    def predict_tile(self, prompt: str, x: int, y: int) -> int:
        logits = self.net(world_features(prompt, x, y)).tolist()[0]
        return int(max(range(len(logits)), key=lambda index: logits[index]))


def world_features(prompt: str, x: int, y: int) -> Tensor:
    prompt_seed = sum(prompt.encode("utf-8")) % 997
    values = [
        x / 12.0,
        y / 12.0,
        sin(x * 0.7),
        sin(y * 0.7),
        ((prompt_seed % 17) / 8.0) - 1.0,
        ((prompt_seed % 29) / 14.0) - 1.0,
        1.0 if "water" in prompt.lower() or "ocean" in prompt.lower() else 0.0,
        1.0 if "ruin" in prompt.lower() or "city" in prompt.lower() else 0.0,
    ]
    return tensor([values])


def target_tile(prompt: str, x: int, y: int) -> int:
    text = prompt.lower()
    noise = int(abs(sin((x * 31 + y * 17 + sum(prompt.encode("utf-8"))) * 0.13)) * 1000)
    if "water" in text and (x + y + noise) % 5 in {0, 1}:
        return 1
    if "mountain" in text and (x * x + y * y + noise) % 7 in {0, 3}:
        return 2
    if "ruin" in text or "city" in text:
        if (x - y + noise) % 9 in {0, 1}:
            return 4
    if "glow" in text or "magic" in text:
        if (x + 2 * y + noise) % 11 == 0:
            return 3
    return noise % 3 if noise % 13 == 0 else 0


def training_samples(prompt: str, radius: int = 4) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            features.append(world_features(prompt, x, y).tolist()[0])
            labels.append(target_tile(prompt, x, y))
    return features, labels


def render_viewport(
    model: LiveWorldModel,
    prompt: str,
    player: Player,
    *,
    radius: int = 4,
) -> dict:
    rows: list[list[dict[str, object]]] = []
    for y in range(player.y - radius, player.y + radius + 1):
        row: list[dict[str, object]] = []
        for x in range(player.x - radius, player.x + radius + 1):
            tile_id = model.predict_tile(prompt, x, y)
            tile = TILES[tile_id]
            row.append(
                {
                    "x": x,
                    "y": y,
                    "tile": tile_id,
                    "char": "@" if x == player.x and y == player.y else tile["char"],
                    "name": "player" if x == player.x and y == player.y else tile["name"],
                    "color": "#ffffff" if x == player.x and y == player.y else tile["color"],
                }
            )
        rows.append(row)
    return {
        "prompt": prompt,
        "player": {"x": player.x, "y": player.y},
        "radius": radius,
        "tiles": rows,
        "legend": {str(key): value for key, value in TILES.items()},
    }


def move(player: Player, direction: str) -> Player:
    direction = direction.lower()
    if direction == "w":
        return Player(player.x, player.y - 1)
    if direction == "s":
        return Player(player.x, player.y + 1)
    if direction == "a":
        return Player(player.x - 1, player.y)
    if direction == "d":
        return Player(player.x + 1, player.y)
    return player


def render_ascii(viewport: dict) -> str:
    return "\n".join("".join(str(tile["char"]) for tile in row) for row in viewport["tiles"])


def tile_histogram(rows: Iterable[Iterable[dict[str, object]]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for row in rows:
        for tile in row:
            name = str(tile["name"])
            hist[name] = hist.get(name, 0) + 1
    return hist
