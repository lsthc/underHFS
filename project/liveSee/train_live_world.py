from __future__ import annotations

import json
import sys
from pathlib import Path

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

from live_world import LiveWorldModel, block_histogram, render_ascii, render_chunk, spawn_player, training_samples


def run_smoke(*, steps: int = 18, write_artifacts: bool = False) -> dict:
    prompt = "glowing water ruins"
    model = LiveWorldModel()
    features, labels = training_samples(prompt)
    x = tensor(features)
    y = tensor(labels)
    opt = SGD(model.parameters(), lr=0.05)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    chunk = render_chunk(model, prompt, spawn_player(prompt), radius=4)
    report = {
        "project": "liveSee",
        "task": "minecraft_like_ai_live_world_generation_and_play",
        "prompt": prompt,
        "controls": "WASD + Q/E or arrow keys",
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mode": chunk["mode"],
        "player": chunk["player"],
        "ascii_view": render_ascii(chunk),
        "block_count": len(chunk["blocks"]),
        "block_histogram": block_histogram(chunk["blocks"]),
    }
    if write_artifacts:
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
