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


def encode_command(command: str, features: int = 10):
    raw = command.lower().encode("utf-8")[:features]
    values = [(byte % 32) / 16.0 - 1.0 for byte in raw]
    values.extend([0.0] * (features - len(values)))
    return tensor([values])


def target_world(command: str):
    text = command.lower()
    goal_x = 1.0 if "east" in text else -1.0 if "west" in text else 0.0
    goal_y = 1.0 if "north" in text else -1.0 if "south" in text else 0.0
    resource = 1.0 if "forest" in text or "water" in text else 0.25
    hazard = 1.0 if "storm" in text or "lava" in text else 0.0
    return tensor([[0.0, 0.0, goal_x, goal_y, resource, hazard]])


def build_model() -> Sequential:
    return Sequential(Linear(10, 12), ReLU(), Linear(12, 6))


def run_smoke(*, steps: int = 10, write_artifacts: bool = False) -> dict:
    command = "move north east toward forest"
    model = build_model()
    opt = SGD(model.parameters(), lr=0.04)
    x = encode_command(command)
    y = target_world(command)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    report = {
        "project": "text2world",
        "command": command,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "world_vector": model(x).tolist()[0],
        "schema": ["agent_x", "agent_y", "goal_x", "goal_y", "resource", "hazard"],
    }
    if write_artifacts:
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
