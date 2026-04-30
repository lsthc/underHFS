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
        if lr < 0:
            raise ValueError("lr must be non-negative")
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
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("AdamW betas must be in the range [0, 1)")
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [
            zeros(param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
            for param in self.params
        ]
        self.v = [
            zeros(param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
            for param in self.params
        ]

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            self._step_param(index, param, beta1, beta2)

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": [state.tolist() for state in self.m],
            "v": [state.tolist() for state in self.v],
        }

    def load_state_dict(self, state: dict) -> None:
        if len(state["m"]) != len(self.params) or len(state["v"]) != len(self.params):
            raise ValueError("optimizer state does not match parameter count")
        self.step_count = int(state["step_count"])
        self.lr = float(state["lr"])
        self.betas = (float(state["betas"][0]), float(state["betas"][1]))
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.m = [
            Tensor(value, shape=param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
            for value, param in zip(state["m"], self.params, strict=True)
        ]
        self.v = [
            Tensor(value, shape=param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
            for value, param in zip(state["v"], self.params, strict=True)
        ]

    def _step_param(self, index: int, param: Tensor, beta1: float, beta2: float) -> None:
        grad = param.grad + (param * self.weight_decay)
        self.m[index] = self.m[index] * beta1 + grad * (1 - beta1)
        self.v[index] = self.v[index] * beta2 + (grad * grad) * (1 - beta2)
        m_hat = self.m[index] / (1 - beta1**self.step_count)
        v_hat = self.v[index] / (1 - beta2**self.step_count)
        update_values = [
            m / (sqrt(v) + self.eps) for m, v in zip(m_hat._storage, v_hat._storage, strict=True)
        ]
        update = Tensor(update_values, shape=param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
        param.add_(update * -self.lr)


class FusedAdamW(AdamW):
    """AdamW with a fused-style per-parameter update loop.

    The current implementation keeps Python orchestration, but it preserves
    device/dtype state and gives the public optimizer surface a concrete,
    tested behavior while native fused CUDA kernels are brought up.
    """

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        active = [
            (index, param)
            for index, param in enumerate(self.params)
            if param.grad is not None
        ]
        for index, param in active:
            self._step_param(index, param, beta1, beta2)


@dataclass
class ZeroShard:
    rank: int = 0
    world_size: int = 1


class ZeroAwareAdamW(AdamW):
    def __init__(self, params: Iterable[Tensor], shard: ZeroShard | None = None, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.shard = shard or ZeroShard()
