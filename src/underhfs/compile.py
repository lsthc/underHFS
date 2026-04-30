from __future__ import annotations

from dataclasses import dataclass, field, replace
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


GuardSignature = tuple[tuple[str, tuple[int, ...], str, str, str], ...]


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
class CompiledKernel:
    name: str
    kind: FusionKind
    nodes: tuple[str, ...]
    backend: str
    executable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "nodes": list(self.nodes),
            "backend": self.backend,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class ExecutablePlan:
    graph: GraphIR
    kernels: tuple[CompiledKernel, ...]
    fallback_backend: str = "eager-python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "fallback_backend": self.fallback_backend,
        }


@dataclass(frozen=True)
class CompileCacheInfo:
    hits: int = 0
    misses: int = 0
    specializations: int = 0

    def to_dict(self) -> dict[str, int]:
        return self.__dict__


@dataclass(frozen=True)
class CompileCacheEntry:
    signature: GuardSignature
    report: CompileReport
    hits: int = 0


@dataclass(frozen=True)
class CompileReport:
    policy: CompilePolicy
    graph: GraphIR
    guards: tuple[Guard, ...]
    fusion_groups: tuple[FusionGroup, ...]
    specialization_key: GuardSignature = ()
    cache_hit: bool = False
    cache_info: CompileCacheInfo = field(default_factory=CompileCacheInfo)
    plan: ExecutablePlan | None = None

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
            "specialization_key": [
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "layout": layout,
                }
                for name, shape, dtype, device, layout in self.specialization_key
            ],
            "cache_hit": self.cache_hit,
            "cache_info": self.cache_info.to_dict(),
            "plan": None if self.plan is None else self.plan.to_dict(),
        }


def compile(function: F | None = None, *, policy: CompilePolicy | None = None):
    active_policy = policy or CompilePolicy()

    def decorate(fn: F) -> F:
        cache: dict[GuardSignature, CompileCacheEntry] = {}
        hits = 0
        misses = 0

        def current_cache_info() -> CompileCacheInfo:
            return CompileCacheInfo(hits=hits, misses=misses, specializations=len(cache))

        @wraps(fn)
        def run(*args, **kwargs):
            nonlocal hits, misses
            signature = _guard_signature(args, kwargs)
            if active_policy.enabled and active_policy.guard_specialization and signature in cache:
                result = fn(*args, **kwargs)
                entry = cache[signature]
                hits += 1
                cache[signature] = replace(entry, hits=entry.hits + 1)
                report = replace(
                    entry.report,
                    cache_hit=True,
                    cache_info=current_cache_info(),
                )
                setattr(run, "_underhfs_last_compile_report", report)
                setattr(run, "_underhfs_last_graph", report.graph)
                return result

            result = fn(*args, **kwargs)
            misses += 1
            report = analyze_execution(result, args=args, kwargs=kwargs, policy=active_policy)
            specializations = (
                len(cache) + 1
                if active_policy.enabled and active_policy.guard_specialization
                else len(cache)
            )
            report = replace(
                report,
                specialization_key=signature,
                cache_hit=False,
                cache_info=CompileCacheInfo(hits=hits, misses=misses, specializations=specializations),
            )
            if active_policy.enabled and active_policy.guard_specialization:
                cache[signature] = CompileCacheEntry(signature=signature, report=report)
            setattr(run, "_underhfs_last_compile_report", report)
            setattr(run, "_underhfs_last_graph", report.graph)
            return result

        setattr(run, "_underhfs_compile_policy", active_policy)
        setattr(run, "_underhfs_last_compile_report", None)
        setattr(run, "_underhfs_compile_cache", cache)
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
    plan = lower_to_plan(graph, fusion_groups)
    return CompileReport(
        active_policy,
        graph,
        guards,
        fusion_groups,
        specialization_key=_guards_to_signature(guards),
        plan=plan,
    )


def explain(fn: Callable[..., Any], *args: Any, policy: CompilePolicy | None = None, **kwargs: Any) -> CompileReport:
    result = fn(*args, **kwargs)
    return analyze_execution(result, args=args, kwargs=kwargs, policy=policy)


def lower_to_plan(graph: GraphIR, fusion_groups: tuple[FusionGroup, ...]) -> ExecutablePlan:
    kernels = tuple(
        CompiledKernel(
            name=f"kernel_{index}_{group.kind.value}",
            kind=group.kind,
            nodes=group.nodes,
            backend="eager-fused-plan",
            executable=True,
        )
        for index, group in enumerate(fusion_groups)
    )
    return ExecutablePlan(graph=graph, kernels=kernels)


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


def _guard_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> GuardSignature:
    return _guards_to_signature(_guards(args, dict(sorted(kwargs.items()))))


def _guards_to_signature(guards: tuple[Guard, ...]) -> GuardSignature:
    return tuple((guard.name, guard.shape, guard.dtype, guard.device, guard.layout) for guard in guards)


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
