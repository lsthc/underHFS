from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from underhfs.tensor import Tensor, zeros


class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            param.add_(param.grad * -self.lr)


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [zeros(param.shape) for param in self.params]
        self.v = [zeros(param.shape) for param in self.params]

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad + (param * self.weight_decay)
            self.m[index] = self.m[index] * beta1 + grad * (1 - beta1)
            self.v[index] = self.v[index] * beta2 + (grad * grad) * (1 - beta2)
            m_hat = self.m[index] / (1 - beta1**self.step_count)
            v_hat = self.v[index] / (1 - beta2**self.step_count)
            update_values = [
                m / (sqrt(v) + self.eps) for m, v in zip(m_hat._storage, v_hat._storage, strict=True)
            ]
            param.add_(Tensor(update_values, shape=param.shape) * -self.lr)


class FusedAdamW(AdamW):
    """API-compatible placeholder for the future fused CUDA implementation."""


@dataclass
class ZeroShard:
    rank: int = 0
    world_size: int = 1


class ZeroAwareAdamW(AdamW):
    def __init__(self, params: Iterable[Tensor], shard: ZeroShard | None = None, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.shard = shard or ZeroShard()
