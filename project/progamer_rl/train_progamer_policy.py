from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from underhfs.functional import cross_entropy
from underhfs.nn import Linear, ReLU, Sequential
from underhfs.optim import SGD
from underhfs.tensor import tensor


ACTIONS = ["up", "down", "left", "right"]


def gridworld_states():
    goal = (2, 2)
    states: list[list[float]] = []
    labels: list[int] = []
    for y in range(3):
        for x in range(3):
            states.append([x / 2.0, y / 2.0, goal[0] / 2.0, goal[1] / 2.0])
            if x < goal[0]:
                labels.append(ACTIONS.index("right"))
            elif y < goal[1]:
                labels.append(ACTIONS.index("down"))
            else:
                labels.append(ACTIONS.index("up"))
    return states, labels


def build_model() -> Sequential:
    return Sequential(Linear(4, 8), ReLU(), Linear(8, len(ACTIONS)))


def run_smoke(*, steps: int = 16, write_artifacts: bool = False) -> dict:
    states, labels = gridworld_states()
    model = build_model()
    opt = SGD(model.parameters(), lr=0.08)
    x = tensor(states)
    y = tensor(labels)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    logits = model(x).tolist()
    policy = [ACTIONS[row.index(max(row))] for row in logits]
    report = {
        "project": "progamer_rl",
        "environment": "3x3 gridworld imitation seed",
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "policy": policy,
        "oracle": [ACTIONS[index] for index in labels],
    }
    if write_artifacts:
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
