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
    native_required: bool = False
    allow_fallback: bool = True
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

    def dispatch(self, *inputs: Any, op: str | None = None, scale: float | None = None, causal: bool = False) -> Any:
        if not self.executable:
            raise RuntimeError(f"{self.name} is not executable for backend {self.backend}")
        if self.backend == "native-cuda-attention":
            return _dispatch_native_cuda_attention(*inputs, scale=scale, causal=causal)
        if self.backend == "native-cuda-fused":
            return _dispatch_native_cuda_fused(*inputs, op=op)
        if self.backend == "eager-fused-plan":
            return _dispatch_eager_fused(*inputs, op=op)
        raise RuntimeError(f"{self.backend} dispatch is not implemented")

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
                "native_required": self.policy.native_required,
                "allow_fallback": self.policy.allow_fallback,
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
    plan = lower_to_plan(graph, fusion_groups, policy=active_policy)
    if active_policy.native_required and any(not kernel.backend.startswith("native-") for kernel in plan.kernels):
        raise RuntimeError("compile native mode found unsupported fusion groups; fallback is disabled")
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


def lower_to_plan(
    graph: GraphIR,
    fusion_groups: tuple[FusionGroup, ...],
    *,
    policy: CompilePolicy | None = None,
) -> ExecutablePlan:
    active_policy = policy or CompilePolicy()
    kernels = tuple(_lowered_kernel(index, graph, group, active_policy) for index, group in enumerate(fusion_groups))
    if not active_policy.allow_fallback and any(not kernel.backend.startswith("native-") for kernel in kernels):
        raise RuntimeError("compile fallback is disabled, but a fusion group has no native backend")
    return ExecutablePlan(graph=graph, kernels=kernels)


def _lowered_kernel(index: int, graph: GraphIR, group: FusionGroup, policy: CompilePolicy) -> CompiledKernel:
    backend = _lowered_backend(graph, group, policy)
    return CompiledKernel(
        name=f"kernel_{index}_{group.kind.value}",
        kind=group.kind,
        nodes=group.nodes,
        backend=backend,
        executable=backend != "unsupported-native",
    )


def _lowered_backend(graph: GraphIR, group: FusionGroup, policy: CompilePolicy) -> str:
    nodes = [node for node in graph.nodes if node.name in group.nodes]
    if not nodes:
        return "eager-fused-plan"
    if group.kind is FusionKind.ATTENTION:
        if _native_cuda_attention_supported(nodes):
            return "native-cuda-attention"
        return "unsupported-native" if policy.native_required or not policy.allow_fallback else "eager-fused-plan"
    if group.kind in {FusionKind.ELEMENTWISE, FusionKind.REDUCTION}:
        if all(node.device.startswith("cuda") and node.dtype == "fp32" for node in nodes):
            return "native-cuda-fused"
        return "unsupported-native" if policy.native_required or not policy.allow_fallback else "eager-fused-plan"
    return "eager-fused-plan"


def _native_cuda_attention_supported(nodes: list[IRNode]) -> bool:
    shapes = [node.shape for node in nodes if node.shape]
    return (
        bool(shapes)
        and all(len(shape) == 2 for shape in shapes)
        and all(node.device.startswith("cuda") and node.dtype == "fp32" for node in nodes)
    )


def _dispatch_native_cuda_attention(*inputs: Any, scale: float | None, causal: bool) -> Any:
    if len(inputs) != 3:
        raise ValueError("native CUDA attention dispatch expects q, k, and v tensors")
    q, k, v = inputs
    _require_supported_cuda_tensor(q, "q")
    _require_supported_cuda_tensor(k, "k")
    _require_supported_cuda_tensor(v, "v")
    if q.shape != k.shape or q.shape != v.shape or len(q.shape) != 2:
        raise ValueError("native CUDA attention dispatch expects q/k/v shape [tokens, features]")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("native CUDA attention dispatch requires contiguous q/k/v tensors")
    from underhfs.native import require_native
    from underhfs.tensor import Tensor

    core = require_native()
    if not bool(getattr(core, "cuda_enabled", False)) or not hasattr(core, "cuda_attention_f32"):
        raise RuntimeError("native CUDA attention dispatch requires _core.cuda_attention_f32")
    tokens, features = q.shape
    actual_scale = (features**-0.5) if scale is None else float(scale)
    values = core.cuda_attention_f32(
        [float(value) for value in q._flat_values()],
        [float(value) for value in k._flat_values()],
        [float(value) for value in v._flat_values()],
        int(tokens),
        int(features),
        actual_scale,
        bool(causal),
    )
    out = Tensor(list(values), shape=q.shape, dtype=q.dtype, device=q.device, layout=q.layout)
    out._attach_cuda_storage()
    out.backend = "native_cuda_attention"
    return out


def _dispatch_native_cuda_fused(*inputs: Any, op: str | None) -> Any:
    if op not in {"add", "mul", "sum"}:
        raise ValueError("native CUDA fused dispatch currently supports op='add', op='mul', or op='sum'")
    if op in {"add", "mul"}:
        if len(inputs) != 2:
            raise ValueError(f"native CUDA fused {op} dispatch expects two tensors")
        left, right = inputs
        _require_supported_cuda_tensor(left, "left")
        _require_supported_cuda_tensor(right, "right")
        return left + right if op == "add" else left * right
    if len(inputs) != 1:
        raise ValueError("native CUDA fused sum dispatch expects one tensor")
    (value,) = inputs
    _require_supported_cuda_tensor(value, "value")
    return value.sum()


def _dispatch_eager_fused(*inputs: Any, op: str | None) -> Any:
    if op == "add" and len(inputs) == 2:
        return inputs[0] + inputs[1]
    if op == "mul" and len(inputs) == 2:
        return inputs[0] * inputs[1]
    if op == "sum" and len(inputs) == 1:
        return inputs[0].sum()
    raise RuntimeError("eager fused plan dispatch needs an explicit supported op")


def _require_supported_cuda_tensor(value: Any, name: str) -> None:
    if not _looks_like_tensor(value):
        raise TypeError(f"{name} must be an underHFS Tensor")
    if str(getattr(value, "device", "")) != "cuda:0":
        raise ValueError(f"{name} must be on cuda:0")
    if getattr(getattr(value, "dtype", None), "value", "") != "fp32":
        raise ValueError(f"{name} must be fp32")
    if getattr(getattr(value, "layout", None), "value", "") != "dense":
        raise ValueError(f"{name} must be dense")


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
