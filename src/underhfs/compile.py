from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


class FusionKind(str, Enum):
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    ATTENTION = "attention"


@dataclass
class CompilePolicy:
    enabled: bool = True
    partial_dynamic_shapes: bool = True
    guard_specialization: bool = True
    fusion: tuple[FusionKind, ...] = field(
        default_factory=lambda: (FusionKind.ELEMENTWISE, FusionKind.REDUCTION, FusionKind.ATTENTION)
    )


@dataclass
class IRNode:
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass
class GraphIR:
    nodes: list[IRNode] = field(default_factory=list)

    def add(self, op: str, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
        self.nodes.append(IRNode(op, inputs, outputs))


def compile(function: F | None = None, *, policy: CompilePolicy | None = None):
    active_policy = policy or CompilePolicy()

    def decorate(fn: F) -> F:
        setattr(fn, "_underhfs_compile_policy", active_policy)
        return fn

    if function is None:
        return decorate
    return decorate(function)
