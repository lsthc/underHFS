from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from underhfs import functional as F
from underhfs.nn import Linear, ReLU, Sequential
from underhfs.optim import AdamW
from underhfs.serialization import save_checkpoint
from underhfs.tensor import tensor


BOARD_SIZE = 9
POINTS = BOARD_SIZE * BOARD_SIZE
PASS_MOVE = POINTS
FEATURES = POINTS * 4
PROJECT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT / "artifacts"
REPORTS = PROJECT / "reports"


@dataclass(frozen=True)
class Sample:
    features: list[float]
    policy: int
    value: float
    strategy: str
    board: tuple[int, ...]


class BetaGoAgent:
    def __init__(self) -> None:
        self.trunk = Sequential(Linear(FEATURES, 32), ReLU(), Linear(32, 16), ReLU())
        self.policy_head = Linear(16, POINTS + 1)
        self.value_head = Linear(16, 1)

    def __call__(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def parameters(self):
        yield from self.trunk.parameters()
        yield from self.policy_head.parameters()
        yield from self.value_head.parameters()

    def state_dict(self) -> dict[str, list | float]:
        state = {}
        for prefix, module in (("trunk", self.trunk), ("policy", self.policy_head), ("value", self.value_head)):
            for name, value in module.state_dict().items():
                state[f"{prefix}.{name}"] = value
        return state


def neighbors(index: int) -> list[int]:
    row, col = divmod(index, BOARD_SIZE)
    out = []
    if row > 0:
        out.append(index - BOARD_SIZE)
    if row + 1 < BOARD_SIZE:
        out.append(index + BOARD_SIZE)
    if col > 0:
        out.append(index - 1)
    if col + 1 < BOARD_SIZE:
        out.append(index + 1)
    return out


def group_and_liberties(board: tuple[int, ...], start: int) -> tuple[set[int], set[int]]:
    color = board[start]
    group = {start}
    liberties: set[int] = set()
    stack = [start]
    while stack:
        point = stack.pop()
        for nb in neighbors(point):
            if board[nb] == 0:
                liberties.add(nb)
            elif board[nb] == color and nb not in group:
                group.add(nb)
                stack.append(nb)
    return group, liberties


def legal_moves(board: tuple[int, ...], color: int) -> list[int]:
    legal = []
    opponent = -color
    for point, value in enumerate(board):
        if value != 0:
            continue
        trial = list(board)
        trial[point] = color
        captures = False
        for nb in neighbors(point):
            if board[nb] == opponent:
                _, libs = group_and_liberties(tuple(trial), nb)
                if not libs:
                    captures = True
                    break
        _, own_libs = group_and_liberties(tuple(trial), point)
        if own_libs or captures:
            legal.append(point)
    return legal or [PASS_MOVE]


def move_score(board: tuple[int, ...], color: int, move: int) -> tuple[float, str]:
    if move == PASS_MOVE:
        return -1.0, "pass"
    opponent = -color
    row, col = divmod(move, BOARD_SIZE)
    score = 0.0
    tags: list[str] = []

    for nb in neighbors(move):
        if board[nb] == opponent:
            group, libs = group_and_liberties(board, nb)
            if move in libs and len(libs) == 1:
                score += 9.0 + len(group)
                tags.append("capture")
            elif move in libs and len(libs) == 2:
                score += 3.0
                tags.append("atari")
        elif board[nb] == color:
            group, libs = group_and_liberties(board, nb)
            if move in libs and len(libs) == 1:
                score += 8.0 + 0.5 * len(group)
                tags.append("defend_atari")
            elif move in libs:
                score += 1.8
                tags.append("connect")

    distance_to_edge = min(row, col, BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col)
    if (row, col) in {(2, 2), (2, 6), (6, 2), (6, 6)}:
        score += 2.4
        tags.append("corner_framework")
    elif distance_to_edge == 2:
        score += 1.2
        tags.append("side_extension")
    elif 3 <= row <= 5 and 3 <= col <= 5:
        score += 1.0
        tags.append("center_influence")

    friendly = sum(1 for nb in neighbors(move) if board[nb] == color)
    enemy = sum(1 for nb in neighbors(move) if board[nb] == opponent)
    score += 0.45 * friendly + 0.25 * enemy
    return score, tags[0] if tags else "shape_balance"


def encode(board: tuple[int, ...], color: int) -> list[float]:
    opponent = -color
    legal = set(legal_moves(board, color))
    features = []
    for point, value in enumerate(board):
        features.append(1.0 if value == color else 0.0)
    for point, value in enumerate(board):
        features.append(1.0 if value == opponent else 0.0)
    for point in range(POINTS):
        features.append(1.0 if point in legal else 0.0)
    features.extend([float(color)] * POINTS)
    return features


def generate_position(seed: int) -> tuple[tuple[int, ...], int]:
    board = [0] * POINTS
    stones = 10 + (seed % 18)
    color = 1
    cursor = (seed * 17 + 11) % POINTS
    for step in range(stones):
        for attempt in range(POINTS):
            point = (cursor + attempt * (7 + seed % 5) + step * 13) % POINTS
            if board[point] == 0:
                board[point] = color
                break
        color *= -1
        cursor = (cursor * 19 + 23) % POINTS
    return tuple(board), 1 if seed % 2 == 0 else -1


def generate_samples(count: int) -> list[Sample]:
    samples = []
    for seed in range(count):
        templated = strategy_template(seed)
        if templated is None:
            board, color = generate_position(seed)
            legal = legal_moves(board, color)
            ranked = [(move_score(board, color, move), move) for move in legal]
            ranked.sort(key=lambda item: (item[0][0], -item[1]), reverse=True)
            (best_score, strategy), move = ranked[0]
        else:
            board, color, move, strategy, best_score = templated
        occupied_balance = sum(board) * color / max(1, POINTS)
        value = max(-1.0, min(1.0, occupied_balance + best_score / 14.0))
        samples.append(Sample(encode(board, color), move, value, strategy, board))
    return samples


def strategy_template(seed: int) -> tuple[tuple[int, ...], int, int, str, float] | None:
    strategy = seed % 8
    board = [0] * POINTS
    color = 1
    if strategy == 0:
        move = 40
        board[31] = board[39] = -1
        board[49] = 1
        return tuple(board), color, move, "capture", 12.0
    if strategy == 1:
        move = 40
        board[39] = board[49] = 1
        board[31] = board[41] = -1
        return tuple(board), color, move, "defend_atari", 11.0
    if strategy == 2:
        move = 40
        board[39] = board[41] = 1
        board[31] = -1
        return tuple(board), color, move, "connect", 8.0
    if strategy == 3:
        move = 20
        board[19] = 1
        board[11] = -1
        return tuple(board), color, move, "corner_framework", 7.0
    if strategy == 4:
        move = 22
        board[21] = 1
        board[13] = -1
        return tuple(board), color, move, "side_extension", 6.5
    if strategy == 5:
        move = 40
        board[30] = board[50] = 1
        board[32] = board[48] = -1
        return tuple(board), color, move, "center_influence", 6.0
    if strategy == 6:
        move = 31
        board[40] = -1
        board[39] = board[49] = 1
        return tuple(board), color, move, "atari", 7.5
    if strategy == 7:
        move = 58
        board[57] = 1
        board[49] = -1
        board[67] = 1
        return tuple(board), color, move, "shape_balance", 5.5
    return None


def batch(samples: list[Sample], start: int, size: int) -> list[Sample]:
    return [samples[(start + i) % len(samples)] for i in range(size)]


def evaluate(model: BetaGoAgent, samples: list[Sample]) -> dict[str, float | dict[str, float]]:
    correct = 0
    top3 = 0
    value_abs = 0.0
    by_strategy: dict[str, list[int]] = {}
    for sample in samples:
        logits, value = model(tensor([sample.features]))
        scores = [(logits._value_at((0, i)), i) for i in range(POINTS + 1)]
        scores.sort(reverse=True)
        pred = scores[0][1]
        if pred == sample.policy:
            correct += 1
        if sample.policy in [move for _, move in scores[:3]]:
            top3 += 1
        value_abs += abs(value._value_at((0, 0)) - sample.value)
        by_strategy.setdefault(sample.strategy, [0, 0])
        by_strategy[sample.strategy][1] += 1
        if pred == sample.policy:
            by_strategy[sample.strategy][0] += 1
    return {
        "policy_top1": correct / len(samples),
        "policy_top3": top3 / len(samples),
        "value_mae": value_abs / len(samples),
        "strategy_accuracy": {name: wins / total for name, (wins, total) in sorted(by_strategy.items())},
    }


def train() -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    train_samples = generate_samples(96)
    eval_samples = generate_samples(40)[-32:]
    model = BetaGoAgent()
    opt = AdamW(model.parameters(), lr=0.015, weight_decay=0.0001)
    history = []
    batch_size = 8
    steps = 60
    for step in range(steps):
        current = batch(train_samples, step * batch_size, batch_size)
        x = tensor([sample.features for sample in current])
        policy_target = tensor([sample.policy for sample in current])
        value_target = tensor([[sample.value] for sample in current])
        logits, value = model(x)
        policy_loss = F.cross_entropy(logits, policy_target)
        value_loss = F.mse_loss(value, value_target)
        loss = policy_loss + value_loss * 0.35
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            metrics = evaluate(model, eval_samples)
            metrics["step"] = step
            metrics["loss"] = loss.item()
            history.append(metrics)
            print(json.dumps(metrics, indent=2))
    final_metrics = evaluate(model, eval_samples)
    state_path = ARTIFACTS / "betago_agent_state.json"
    save_checkpoint(
        state_path,
        state=model.state_dict(),
        metadata={"board_size": BOARD_SIZE, "features": FEATURES, "moves": POINTS + 1, "training": "synthetic_strategy_supervision"},
    )
    payload = {
        "board_size": BOARD_SIZE,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "steps": steps,
        "batch_size": batch_size,
        "final_metrics": final_metrics,
        "history": history,
        "strategies": sorted({sample.strategy for sample in train_samples}),
        "artifact": str(state_path),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    return payload


def write_report(metrics: dict) -> None:
    final = metrics["final_metrics"]
    strategy_rows = "\n".join(
        f"| {name} | {acc:.3f} |"
        for name, acc in final["strategy_accuracy"].items()
    )
    report = f"""# BetaGo underHFS Agent Report

## Scope

This test project trained BetaGo, a small 9x9 Baduk policy/value agent using underHFS
Tensor, autograd, nn modules, optimizer, and serialization. The project does not
claim professional Go strength or real mastery of divine moves. Instead, it
tests whether underHFS can run an end-to-end agent-training workflow over
rule-aware, strategy-labeled Baduk positions.

## Training Setup

- Board: 9x9
- Input features: own stones, opponent stones, legal moves, side-to-move plane
- Model: Linear(324, 32) -> ReLU -> Linear(32, 16) -> ReLU with policy/value heads
- Optimizer: underHFS AdamW
- Training samples: {metrics['train_samples']}
- Evaluation samples: {metrics['eval_samples']}
- Steps: {metrics['steps']}
- Batch size: {metrics['batch_size']}

## Strategy Labels

Synthetic labels include capture, atari, defend-atari, connection, side extension,
corner framework, center influence, and shape balance. These are heuristic
training targets, not game-record-supervised professional moves.

## Final Metrics

- Policy top-1 accuracy: {final['policy_top1']:.3f}
- Policy top-3 accuracy: {final['policy_top3']:.3f}
- Value MAE: {final['value_mae']:.3f}

## Accuracy By Strategy

| Strategy | Top-1 Accuracy |
| --- | --- |
{strategy_rows}

## Artifacts

- Model checkpoint: `{metrics['artifact']}`
- Metrics JSON: `project/BetaGo/artifacts/metrics.json`

## Limitations

- No self-play search, MCTS, SGF/pro game corpus, life-and-death solver, ko/superko
  history, komi, territory scoring, or full board-size 19x19 training is included.
- "God move" behavior is represented only as selecting the highest-scoring move
  under the handcrafted tactical/shape heuristic. It is not evidence of superhuman
  Go ability.
- The purpose is to validate underHFS as a local training runtime and produce a
  reproducible BetaGo baseline project for future stronger Baduk experiments.
"""
    (REPORTS / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
