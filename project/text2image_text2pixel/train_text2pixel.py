from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from underhfs.functional import mse_loss
from underhfs.nn import Linear, ReLU, Sequential
from underhfs.optim import SGD
from underhfs.tensor import tensor


def encode_prompt(prompt: str, features: int = 8):
    raw = prompt.encode("utf-8")[:features]
    values = [(byte / 127.5) - 1.0 for byte in raw]
    values.extend([0.0] * (features - len(values)))
    return tensor([values])


def target_pixels(prompt: str):
    seed = sum(prompt.encode("utf-8")) or 1
    values = [((seed * (index + 3)) % 257) / 128.0 - 1.0 for index in range(16)]
    return tensor([values])


def build_model() -> Sequential:
    return Sequential(Linear(8, 16), ReLU(), Linear(16, 16))


def run_smoke(*, steps: int = 8, write_artifacts: bool = False) -> dict:
    prompt = "a small red sun"
    model = build_model()
    opt = SGD(model.parameters(), lr=0.03)
    x = encode_prompt(prompt)
    y = target_pixels(prompt)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    generated = model(x).tolist()[0]
    report = {
        "project": "text2image_text2pixel",
        "prompt": prompt,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "pixel_shape": [1, 4, 4],
        "generated_pixels": [generated[index : index + 4] for index in range(0, 16, 4)],
    }
    if write_artifacts:
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
