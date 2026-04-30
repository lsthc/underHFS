from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from underhfs.tensor import Tensor


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


def jvp(function, primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]):
    raise NotImplementedError(
        "forward-mode JVP is part of the public design surface; native implementation is pending"
    )
