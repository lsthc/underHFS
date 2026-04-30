from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

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
    name: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    shape: tuple[int, ...] = ()
    dtype: str = "unknown"
    device: str = "unknown"


@dataclass
class GraphIR:
    nodes: list[IRNode] = field(default_factory=list)

    def add(
        self,
        *,
        name: str,
        op: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        shape: tuple[int, ...] = (),
        dtype: str = "unknown",
        device: str = "unknown",
    ) -> None:
        self.nodes.append(
            IRNode(
                name=name,
                op=op,
                inputs=inputs,
                outputs=outputs,
                shape=shape,
                dtype=dtype,
                device=device,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [node.__dict__ for node in self.nodes]}


@dataclass(frozen=True)
class Guard:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class FusionGroup:
    kind: FusionKind
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "nodes": list(self.nodes)}


@dataclass(frozen=True)
class CompileReport:
    policy: CompilePolicy
    graph: GraphIR
    guards: tuple[Guard, ...]
    fusion_groups: tuple[FusionGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": {
                "enabled": self.policy.enabled,
                "partial_dynamic_shapes": self.policy.partial_dynamic_shapes,
                "guard_specialization": self.policy.guard_specialization,
                "fusion": [kind.value for kind in self.policy.fusion],
            },
            "graph": self.graph.to_dict(),
            "guards": [guard.to_dict() for guard in self.guards],
            "fusion_groups": [group.to_dict() for group in self.fusion_groups],
        }


def compile(function: F | None = None, *, policy: CompilePolicy | None = None):
    active_policy = policy or CompilePolicy()

    def decorate(fn: F) -> F:
        @wraps(fn)
        def run(*args, **kwargs):
            result = fn(*args, **kwargs)
            report = analyze_execution(result, args=args, kwargs=kwargs, policy=active_policy)
            setattr(run, "_underhfs_last_compile_report", report)
            setattr(run, "_underhfs_last_graph", report.graph)
            return result

        setattr(run, "_underhfs_compile_policy", active_policy)
        setattr(run, "_underhfs_last_compile_report", None)
        return run  # type: ignore[return-value]

    if function is None:
        return decorate
    return decorate(function)


def analyze_execution(
    result: Any,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    policy: CompilePolicy | None = None,
) -> CompileReport:
    active_policy = policy or CompilePolicy()
    graph = GraphIR()
    visited: set[int] = set()

    def visit(value: Any) -> str | None:
        if not _looks_like_tensor(value):
            return None
        node_id = id(value)
        name = f"t{len(visited)}"
        if node_id in visited:
            return f"t{_node_index(node_id, visited_order)}"
        visited.add(node_id)
        visited_order.append(node_id)
        inputs = tuple(parent_name for parent in getattr(value, "_prev", ()) if (parent_name := visit(parent)))
        op = getattr(value, "_op", "") or "leaf"
        graph.add(
            name=name,
            op=op,
            inputs=inputs,
            outputs=(name,),
            shape=tuple(getattr(value, "shape", ())),
            dtype=getattr(getattr(value, "dtype", None), "value", "unknown"),
            device=str(getattr(value, "device", "unknown")),
        )
        return name

    visited_order: list[int] = []
    for value in _flatten_outputs(result):
        visit(value)

    guards = _guards(args, kwargs or {})
    fusion_groups = _fusion_groups(graph, active_policy)
    return CompileReport(active_policy, graph, guards, fusion_groups)


def explain(fn: Callable[..., Any], *args: Any, policy: CompilePolicy | None = None, **kwargs: Any) -> CompileReport:
    result = fn(*args, **kwargs)
    return analyze_execution(result, args=args, kwargs=kwargs, policy=policy)


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "_prev")


def _flatten_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(value.values())
    return (value,)


def _node_index(node_id: int, order: list[int]) -> int:
    try:
        return order.index(node_id)
    except ValueError:
        return -1


def _guards(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Guard, ...]:
    guards: list[Guard] = []
    for index, value in enumerate(args):
        if _looks_like_tensor(value):
            guards.append(_guard(f"arg{index}", value))
    for name, value in kwargs.items():
        if _looks_like_tensor(value):
            guards.append(_guard(name, value))
    return tuple(guards)


def _guard(name: str, value: Any) -> Guard:
    return Guard(
        name=name,
        shape=tuple(getattr(value, "shape", ())),
        dtype=getattr(getattr(value, "dtype", None), "value", "unknown"),
        device=str(getattr(value, "device", "unknown")),
        layout=getattr(getattr(value, "layout", None), "value", "unknown"),
    )


def _fusion_groups(graph: GraphIR, policy: CompilePolicy) -> tuple[FusionGroup, ...]:
    groups: list[FusionGroup] = []
    if FusionKind.ELEMENTWISE in policy.fusion:
        elementwise = tuple(node.name for node in graph.nodes if node.op in {"add", "sub", "mul", "div", "relu", "tanh", "exp", "log", "pow"})
        if len(elementwise) >= 2:
            groups.append(FusionGroup(FusionKind.ELEMENTWISE, elementwise))
    if FusionKind.REDUCTION in policy.fusion:
        reductions = tuple(node.name for node in graph.nodes if node.op in {"sum", "mean"})
        if reductions:
            groups.append(FusionGroup(FusionKind.REDUCTION, reductions))
    if FusionKind.ATTENTION in policy.fusion:
        ops = [node.op for node in graph.nodes]
        if ops.count("matmul") >= 2 and "softmax" in ops:
            groups.append(FusionGroup(FusionKind.ATTENTION, tuple(node.name for node in graph.nodes if node.op in {"matmul", "softmax"})))
    return tuple(groups)
