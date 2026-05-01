from __future__ import annotations

from dataclasses import dataclass
from math import cos, floor, sin
from typing import Iterable

from underhfs.nn import Linear, ReLU, Sequential
from underhfs.tensor import Tensor, tensor


BLOCKS = {
    0: {"name": "air", "color": "#87c8ff", "solid": False},
    1: {"name": "grass", "color": "#55aa55", "solid": True},
    2: {"name": "dirt", "color": "#8b5a2b", "solid": True},
    3: {"name": "stone", "color": "#8c8c8c", "solid": True},
    4: {"name": "water", "color": "#3188d4", "solid": False},
    5: {"name": "glowstone", "color": "#f2c94c", "solid": True},
    6: {"name": "ruin", "color": "#5d6466", "solid": True},
}


@dataclass(frozen=True)
class Player:
    x: float = 0.0
    y: float = 4.0
    z: float = 0.0
    yaw: float = 0.0


class LiveWorldModel:
    def __init__(self) -> None:
        self.net = Sequential(Linear(10, 18), ReLU(), Linear(18, len(BLOCKS)))

    def parameters(self):
        return self.net.parameters()

    def __call__(self, features: Tensor) -> Tensor:
        return self.net(features)

    def predict_surface_block(self, prompt: str, x: int, z: int) -> int:
        y = height_at(prompt, x, z)
        logits = self.net(world_features(prompt, x, y, z)).tolist()[0]
        block = int(max(range(1, len(logits)), key=lambda index: logits[index]))
        return block if BLOCKS[block]["solid"] else 1


def prompt_seed(prompt: str) -> int:
    return sum((index + 1) * byte for index, byte in enumerate(prompt.encode("utf-8"))) % 7919


def height_at(prompt: str, x: int, z: int) -> int:
    seed = prompt_seed(prompt)
    base = 3
    rolling = sin((x + seed % 23) * 0.45) + cos((z - seed % 17) * 0.38)
    mountain = 2 if "mountain" in prompt.lower() or "cliff" in prompt.lower() else 0
    return max(1, min(8, base + int(round(rolling + mountain))))


def world_features(prompt: str, x: int, y: int, z: int) -> Tensor:
    seed = prompt_seed(prompt)
    text = prompt.lower()
    values = [
        x / 16.0,
        y / 8.0,
        z / 16.0,
        sin(x * 0.37),
        cos(z * 0.41),
        sin((x + z) * 0.19),
        ((seed % 31) / 15.0) - 1.0,
        1.0 if "water" in text or "ocean" in text else 0.0,
        1.0 if "ruin" in text or "city" in text else 0.0,
        1.0 if "glow" in text or "magic" in text else 0.0,
    ]
    return tensor([values])


def target_surface_block(prompt: str, x: int, z: int) -> int:
    text = prompt.lower()
    seed = prompt_seed(prompt)
    noise = int(abs(sin((x * 31 + z * 17 + seed) * 0.13)) * 1000)
    if ("water" in text or "ocean" in text) and height_at(prompt, x, z) <= 3 and noise % 5 in {0, 1, 2}:
        return 4
    if "ruin" in text or "city" in text:
        if (x - z + noise) % 11 in {0, 1}:
            return 6
    if "glow" in text or "magic" in text:
        if (x + 2 * z + noise) % 13 == 0:
            return 5
    if height_at(prompt, x, z) >= 6:
        return 3
    return 1


def block_at(model: LiveWorldModel, prompt: str, x: int, y: int, z: int) -> int:
    ground = height_at(prompt, x, z)
    if y > ground:
        return 0
    if y == ground:
        return model.predict_surface_block(prompt, x, z)
    if y >= ground - 2:
        return 2
    return 3


def training_samples(prompt: str, radius: int = 5) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    for z in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            y = height_at(prompt, x, z)
            features.append(world_features(prompt, x, y, z).tolist()[0])
            labels.append(target_surface_block(prompt, x, z))
    return features, labels


def render_chunk(
    model: LiveWorldModel,
    prompt: str,
    player: Player,
    *,
    radius: int = 8,
) -> dict:
    center_x = floor(player.x)
    center_z = floor(player.z)
    blocks: list[dict[str, object]] = []
    for z in range(center_z - radius, center_z + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            ground = height_at(prompt, x, z)
            for y in range(max(0, ground - 3), ground + 1):
                block_id = block_at(model, prompt, x, y, z)
                if block_id == 0:
                    continue
                block = BLOCKS[block_id]
                blocks.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "block": block_id,
                        "name": block["name"],
                        "color": block["color"],
                        "solid": block["solid"],
                    }
                )
    return {
        "prompt": prompt,
        "mode": "minecraft_like_voxel_world",
        "player": player_dict(player),
        "radius": radius,
        "blocks": blocks,
        "legend": {str(key): value for key, value in BLOCKS.items()},
    }


def player_dict(player: Player) -> dict[str, float]:
    return {"x": player.x, "y": player.y, "z": player.z, "yaw": player.yaw}


def spawn_player(prompt: str) -> Player:
    y = float(height_at(prompt, 0, 0) + 2)
    return Player(0.0, y, 0.0, 0.0)


def move(player: Player, direction: str, prompt: str = "", *, step: float = 1.0) -> Player:
    direction = direction.lower()
    yaw = player.yaw
    dx = sin(yaw)
    dz = -cos(yaw)
    sx = cos(yaw)
    sz = sin(yaw)
    x, z = player.x, player.z
    if direction == "w":
        x += dx * step
        z += dz * step
    elif direction == "s":
        x -= dx * step
        z -= dz * step
    elif direction == "a":
        x -= sx * step
        z -= sz * step
    elif direction == "d":
        x += sx * step
        z += sz * step
    elif direction in {"q", "arrowleft"}:
        return Player(player.x, player.y, player.z, player.yaw - 0.25)
    elif direction in {"e", "arrowright"}:
        return Player(player.x, player.y, player.z, player.yaw + 0.25)
    else:
        return player
    ground_y = height_at(prompt, floor(x), floor(z)) + 2 if prompt else player.y
    return Player(round(x, 4), float(ground_y), round(z, 4), yaw)


def render_ascii(chunk: dict) -> str:
    player = chunk["player"]
    px = int(round(float(player["x"])))
    pz = int(round(float(player["z"])))
    radius = int(chunk["radius"])
    top: dict[tuple[int, int], str] = {}
    for block in chunk["blocks"]:
        key = (int(block["x"]), int(block["z"]))
        char = str(block["name"])[0].upper()
        previous = top.get(key)
        if previous is None or int(block["y"]) >= int(previous.split(":", 1)[0]):
            top[key] = f"{block['y']}:{char}"
    rows: list[str] = []
    for z in range(pz - radius, pz + radius + 1):
        row = []
        for x in range(px - radius, px + radius + 1):
            if x == px and z == pz:
                row.append("@")
            else:
                row.append(top.get((x, z), "0:.").split(":", 1)[1])
        rows.append("".join(row))
    return "\n".join(rows)


def block_histogram(blocks: Iterable[dict[str, object]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for block in blocks:
        name = str(block["name"])
        hist[name] = hist.get(name, 0) + 1
    return hist
