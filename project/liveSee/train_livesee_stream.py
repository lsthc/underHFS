from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from underhfs.functional import cross_entropy
from underhfs.nn import Linear
from underhfs.optim import SGD
from underhfs.tensor import tensor


def synthetic_stream_features():
    return [
        [0.05, 0.00, 0.00, 0.0],
        [0.15, 0.10, 0.25, 0.0],
        [0.80, 0.65, 0.50, 1.0],
        [0.95, 0.15, 0.75, 1.0],
    ]


def synthetic_labels():
    return [0, 0, 1, 1]


def run_smoke(*, steps: int = 12, write_artifacts: bool = False) -> dict:
    model = Linear(4, 2)
    opt = SGD(model.parameters(), lr=0.08)
    x = tensor(synthetic_stream_features())
    y = tensor(synthetic_labels())
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    logits = model(x)
    predictions = [row.index(max(row)) for row in logits.tolist()]
    report = {
        "project": "liveSee",
        "task": "synthetic streaming frame classifier",
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "predictions": predictions,
        "labels": synthetic_labels(),
    }
    if write_artifacts:
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
