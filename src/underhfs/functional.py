from __future__ import annotations

from math import exp, log

from underhfs.tensor import Tensor, tensor, zeros


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = input - target
    loss = diff * diff
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unsupported reduction: {reduction}")


def cross_entropy(logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if logits.ndim != 2:
        raise ValueError("cross_entropy fallback expects logits with shape [batch, classes]")
    if target.ndim != 1:
        raise ValueError("cross_entropy fallback expects target with shape [batch]")
    batch, classes = logits.shape
    if target.shape[0] != batch:
        raise ValueError(f"target batch {target.shape[0]} does not match logits batch {batch}")

    losses: list[float] = []
    probs: list[float] = []
    for i in range(batch):
        row = [logits._value_at((i, j)) for j in range(classes)]
        row_max = max(row)
        row_exp = [exp(value - row_max) for value in row]
        denom = sum(row_exp)
        row_probs = [value / denom for value in row_exp]
        probs.extend(row_probs)
        label = int(target._value_at((i,)))
        if label < 0 or label >= classes:
            raise ValueError(f"target class {label} is outside [0, {classes})")
        losses.append(-log(max(row_probs[label], 1e-30)))

    if reduction == "none":
        out = tensor(losses, requires_grad=logits.requires_grad)
        scale = 1.0
    elif reduction == "sum":
        out = tensor(sum(losses), requires_grad=logits.requires_grad)
        scale = 1.0
    elif reduction == "mean":
        out = tensor(sum(losses) / max(1, batch), requires_grad=logits.requires_grad)
        scale = 1.0 / max(1, batch)
    else:
        raise ValueError(f"unsupported reduction: {reduction}")

    out._prev = {logits}
    out._op = "cross_entropy"

    def backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad = zeros(logits.shape)
        if reduction == "none":
            row_scales = [out.grad._value_at((i,)) for i in range(batch)]
        else:
            row_scales = [out.grad.item() * scale for _ in range(batch)]
        for i in range(batch):
            label = int(target._value_at((i,)))
            for j in range(classes):
                value = probs[i * classes + j]
                if j == label:
                    value -= 1.0
                grad._set_at((i, j), value * row_scales[i])
        logits._accumulate_grad(grad)

    out._backward = backward
    return out


def relu(input: Tensor) -> Tensor:
    return input.relu()


def softmax(input: Tensor) -> Tensor:
    return input.softmax()
