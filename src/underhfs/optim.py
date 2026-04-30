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


@dataclass(frozen=True)
class OptimizerKernelStatus:
    name: str
    backend: str
    available: bool
    reason: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "backend": self.backend,
            "available": self.available,
            "reason": self.reason,
        }


def fused_adamw_kernel_status(params: Iterable[Tensor]) -> OptimizerKernelStatus:
    parameters = list(params)
    if not parameters:
        return OptimizerKernelStatus("fused_adamw", "python", True, "no parameters")
    if all(param.device.kind == "cuda" for param in parameters):
        if all(param.dtype.value == "fp32" for param in parameters):
            try:
                from underhfs.native import status, require_native

                native = status()
                core = require_native() if native.cuda_enabled else None
                if native.cuda_enabled and core is not None and hasattr(core, "cuda_fused_adamw_f32"):
                    return OptimizerKernelStatus("fused_adamw", "cuda-native", True, "native CUDA fused AdamW fp32")
            except Exception:
                pass
        return OptimizerKernelStatus(
            "fused_adamw",
            "python",
            True,
            "Python AdamW update preserves CUDA metadata for non-fp32 tensors",
        )
    return OptimizerKernelStatus("fused_adamw", "python", True, "CPU/Python fused update loop")


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
        if self._try_native_fused_adamw_param(index, param, beta1, beta2):
            return
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

    def _try_native_fused_adamw_param(self, index: int, param: Tensor, beta1: float, beta2: float) -> bool:
        if param.grad is None or param.device.kind != "cuda" or param.dtype.value != "fp32":
            return False
        try:
            from underhfs.native import require_native

            core = require_native()
            if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_fused_adamw_f32"):
                return False
            grad = param.grad + (param * self.weight_decay)
            result = core.cuda_fused_adamw_f32(
                [float(value) for value in param._flat_values()],
                [float(value) for value in grad._flat_values()],
                [float(value) for value in self.m[index]._flat_values()],
                [float(value) for value in self.v[index]._flat_values()],
                float(self.lr),
                float(beta1),
                float(beta2),
                float(self.eps),
                0.0,
                int(self.step_count),
            )
        except Exception:
            return False
        param._storage = [float(value) for value in result["param"]]
        param._storage_offset = 0
        param._native_cuda = None
        param._attach_cuda_storage()
        param.backend = "native_cuda"
        param._version_ref[0] += 1
        self.m[index] = Tensor(result["m"], shape=param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
        self.v[index] = Tensor(result["v"], shape=param.shape, dtype=param.dtype, device=param.device, layout=param.layout)
        self.m[index]._attach_cuda_storage()
        self.v[index]._attach_cuda_storage()
        return True


class FusedAdamW(AdamW):
    """AdamW with a fused-style per-parameter update loop.

    The current implementation keeps Python orchestration, but it preserves
    device/dtype state and gives the public optimizer surface a concrete,
    tested behavior while native fused CUDA kernels are brought up.
    """

    def step(self) -> None:
        self.last_kernel_status = fused_adamw_kernel_status(self.params)
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
