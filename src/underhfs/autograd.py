from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from underhfs.tensor import Tensor, tensor


@dataclass
class GradMode:
    enabled: bool = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    return _grad_mode.enabled


@contextmanager
def no_grad() -> Iterator[None]:
    previous = _grad_mode.enabled
    _grad_mode.enabled = False
    try:
        yield
    finally:
        _grad_mode.enabled = previous


def backward(tensor: Tensor, grad: Tensor | None = None) -> None:
    tensor.backward(grad)


def checkpoint(function: Callable[..., Any], *args: Any, preserve_rng_state: bool = True, **kwargs: Any) -> Any:
    result = function(*args, **kwargs)
    _mark_checkpointed(result, function=getattr(function, "__name__", "anonymous"), preserve_rng_state=preserve_rng_state)
    return result


def _mark_checkpointed(value: Any, *, function: str, preserve_rng_state: bool) -> None:
    if isinstance(value, Tensor):
        setattr(
            value,
            "_underhfs_checkpoint",
            {
                "function": function,
                "preserve_rng_state": preserve_rng_state,
                "mode": "eager-recompute-contract",
            },
        )
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _mark_checkpointed(item, function=function, preserve_rng_state=preserve_rng_state)
    if isinstance(value, dict):
        for item in value.values():
            _mark_checkpointed(item, function=function, preserve_rng_state=preserve_rng_state)


def jvp(function: Callable[..., Any], primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]) -> tuple[Any, Any]:
    if len(primals) != len(tangents):
        raise ValueError("primals and tangents must have the same length")
    dual_args = tuple(_DualTensor(primal, tangent) for primal, tangent in zip(primals, tangents, strict=True))
    result = function(*dual_args)
    return _unwrap_dual(result)


class _DualTensor:
    def __init__(self, primal: Tensor, tangent: Tensor) -> None:
        if primal.shape != tangent.shape:
            raise ValueError(f"JVP tangent shape {tangent.shape} does not match primal shape {primal.shape}")
        self.primal = primal
        self.tangent = tangent

    @property
    def shape(self) -> tuple[int, ...]:
        return self.primal.shape

    @property
    def ndim(self) -> int:
        return self.primal.ndim

    @property
    def dtype(self):
        return self.primal.dtype

    @property
    def device(self):
        return self.primal.device

    @property
    def layout(self):
        return self.primal.layout

    @property
    def T(self) -> "_DualTensor":
        return self.transpose()

    def __add__(self, other: Any) -> "_DualTensor":
        rhs = _as_dual(other, like=self.primal)
        return _DualTensor(self.primal + rhs.primal, self.tangent + rhs.tangent)

    def __radd__(self, other: Any) -> "_DualTensor":
        return self + other

    def __sub__(self, other: Any) -> "_DualTensor":
        rhs = _as_dual(other, like=self.primal)
        return _DualTensor(self.primal - rhs.primal, self.tangent - rhs.tangent)

    def __rsub__(self, other: Any) -> "_DualTensor":
        return _as_dual(other, like=self.primal) - self

    def __mul__(self, other: Any) -> "_DualTensor":
        rhs = _as_dual(other, like=self.primal)
        return _DualTensor(self.primal * rhs.primal, self.tangent * rhs.primal + self.primal * rhs.tangent)

    def __rmul__(self, other: Any) -> "_DualTensor":
        return self * other

    def __truediv__(self, other: Any) -> "_DualTensor":
        rhs = _as_dual(other, like=self.primal)
        return _DualTensor(
            self.primal / rhs.primal,
            (self.tangent * rhs.primal - self.primal * rhs.tangent) / (rhs.primal * rhs.primal),
        )

    def __rtruediv__(self, other: Any) -> "_DualTensor":
        return _as_dual(other, like=self.primal) / self

    def __neg__(self) -> "_DualTensor":
        return _DualTensor(-self.primal, -self.tangent)

    def __pow__(self, power: float) -> "_DualTensor":
        return _DualTensor(self.primal**power, self.tangent * (power * (self.primal ** (power - 1))))

    def __matmul__(self, other: Any) -> "_DualTensor":
        rhs = _as_dual(other, like=self.primal)
        return _DualTensor(self.primal @ rhs.primal, self.tangent @ rhs.primal + self.primal @ rhs.tangent)

    def sum(self) -> "_DualTensor":
        return _DualTensor(self.primal.sum(), self.tangent.sum())

    def mean(self) -> "_DualTensor":
        return _DualTensor(self.primal.mean(), self.tangent.mean())

    def relu(self) -> "_DualTensor":
        mask = tensor(
            [1.0 if value > 0 else 0.0 for value in self.primal._flat_values()],
            shape=self.primal.shape,
            dtype=self.primal.dtype,
            device=self.primal.device,
            layout=self.primal.layout,
        )
        return _DualTensor(self.primal.relu(), self.tangent * mask)

    def tanh(self) -> "_DualTensor":
        primal = self.primal.tanh()
        return _DualTensor(primal, self.tangent * (1.0 - primal * primal))

    def exp(self) -> "_DualTensor":
        primal = self.primal.exp()
        return _DualTensor(primal, self.tangent * primal)

    def log(self) -> "_DualTensor":
        return _DualTensor(self.primal.log(), self.tangent / self.primal)

    def transpose(self) -> "_DualTensor":
        return _DualTensor(self.primal.transpose(), self.tangent.transpose())


def _as_dual(value: Any, *, like: Tensor) -> _DualTensor:
    if isinstance(value, _DualTensor):
        return value
    primal = value if isinstance(value, Tensor) else tensor(value, dtype=like.dtype, device=like.device, layout=like.layout)
    return _DualTensor(primal, tensor(0.0, dtype=like.dtype, device=like.device, layout=like.layout))


def _unwrap_dual(value: Any) -> tuple[Any, Any]:
    if isinstance(value, _DualTensor):
        return value.primal, value.tangent
    if isinstance(value, tuple):
        primals = []
        tangents = []
        for item in value:
            primal, tangent = _unwrap_dual(item)
            primals.append(primal)
            tangents.append(tangent)
        return tuple(primals), tuple(tangents)
    if isinstance(value, list):
        primals = []
        tangents = []
        for item in value:
            primal, tangent = _unwrap_dual(item)
            primals.append(primal)
            tangents.append(tangent)
        return primals, tangents
    if isinstance(value, dict):
        primals = {}
        tangents = {}
        for key, item in value.items():
            primal, tangent = _unwrap_dual(item)
            primals[key] = primal
            tangents[key] = tangent
        return primals, tangents
    return value, None
