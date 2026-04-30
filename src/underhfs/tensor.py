from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, log, sqrt, tanh
from typing import Any, Callable, Iterable, Iterator, Sequence


class DType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


class Layout(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    QUANTIZED = "quantized"


@dataclass(frozen=True)
class Device:
    kind: str = "cpu"
    index: int | None = None

    @classmethod
    def parse(cls, value: str | Device) -> Device:
        if isinstance(value, Device):
            return value
        if ":" in value:
            kind, index = value.split(":", 1)
            return cls(kind, int(index))
        return cls(value, None)

    def __str__(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _infer_shape(value: Any) -> tuple[int, ...]:
    if not _is_sequence(value):
        return ()
    if not value:
        return (0,)
    child_shape = _infer_shape(value[0])
    for child in value:
        if _infer_shape(child) != child_shape:
            raise ValueError("ragged tensors are not supported")
    return (len(value), *child_shape)


def _flatten(value: Any) -> list[float]:
    if not _is_sequence(value):
        return [float(value)]
    out: list[float] = []
    for child in value:
        out.extend(_flatten(child))
    return out


def _prod(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def _resolve_shape(shape: Sequence[int], numel: int) -> tuple[int, ...]:
    unknown = [index for index, dim in enumerate(shape) if dim == -1]
    if len(unknown) > 1:
        raise ValueError("only one inferred dimension is allowed")
    resolved = list(shape)
    if unknown:
        known = _prod([dim for dim in resolved if dim != -1])
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {tuple(shape)} for {numel} values")
        resolved[unknown[0]] = numel // known
    if any(dim < 0 for dim in resolved):
        raise ValueError(f"invalid shape {tuple(shape)}")
    return tuple(resolved)


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _iter_indices(shape: Sequence[int]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return
    ranges = [range(dim) for dim in shape]
    indices = [0] * len(shape)

    def visit(axis: int) -> Iterator[tuple[int, ...]]:
        if axis == len(shape):
            yield tuple(indices)
            return
        for value in ranges[axis]:
            indices[axis] = value
            yield from visit(axis + 1)

    yield from visit(0)


def _broadcast_shape(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for a, b in zip(reversed(left), reversed(right), strict=False):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"cannot broadcast shapes {tuple(left)} and {tuple(right)}")
    longer = left if len(left) > len(right) else right
    result.extend(reversed(longer[: abs(len(left) - len(right))]))
    return tuple(reversed(result))


def _broadcast_index(index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    if shape == ():
        return ()
    offset = len(index) - len(shape)
    return tuple(0 if dim == 1 else index[i + offset] for i, dim in enumerate(shape))


def _unbroadcast(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    if grad.shape == shape:
        return grad
    out = zeros(shape, dtype=grad.dtype, device=grad.device, layout=grad.layout)
    for idx in _iter_indices(grad.shape):
        out._add_at(_broadcast_index(idx, shape), grad._value_at(idx))
    return out


class Tensor:
    """A small eager Tensor with reverse-mode autograd.

    This is the portable fallback implementation. Native C++/CUDA backends will
    keep the same Python-facing semantics.
    """

    def __init__(
        self,
        data: Any,
        *,
        shape: Sequence[int] | None = None,
        requires_grad: bool = False,
        dtype: DType | str = DType.FP32,
        device: Device | str = Device(),
        layout: Layout | str = Layout.DENSE,
        _children: Iterable[Tensor] = (),
        _op: str = "",
        _storage: list[float] | None = None,
        _storage_offset: int = 0,
        _strides: Sequence[int] | None = None,
        _version_ref: list[int] | None = None,
    ) -> None:
        inferred = _infer_shape(data) if shape is None else tuple(shape)
        flat = _flatten(data)
        expected = _prod(inferred)
        if expected != len(flat):
            raise ValueError(f"shape {inferred} expects {expected} values, got {len(flat)}")
        self._storage = flat if _storage is None else _storage
        self._storage_offset = _storage_offset
        self.shape = tuple(inferred)
        self.strides = _contiguous_strides(self.shape) if _strides is None else tuple(_strides)
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self.dtype = DType(dtype)
        self.device = Device.parse(device)
        self.layout = Layout(layout)
        self._prev = set(_children)
        self._saved_versions = {child: child.version for child in self._prev}
        self._op = _op
        self._backward: Callable[[], None] = lambda: None
        self._version_ref = [0] if _version_ref is None else _version_ref
        self.backend = "python"
        self._native_cuda = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def version(self) -> int:
        return self._version_ref[0]

    def numel(self) -> int:
        return _prod(self.shape)

    def is_contiguous(self) -> bool:
        return self.strides == _contiguous_strides(self.shape)

    def _flat_values(self) -> list[float]:
        return [self._value_at(index) for index in _iter_indices(self.shape)]

    def clone(self) -> Tensor:
        return Tensor(
            self.tolist(),
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
        )

    def detach(self) -> Tensor:
        return Tensor(
            self.tolist(),
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
        )

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        self.requires_grad = requires_grad
        return self

    def to(
        self,
        device: Device | str | None = None,
        dtype: DType | str | None = None,
        layout: Layout | str | None = None,
    ) -> Tensor:
        target_device = self.device if device is None else Device.parse(device)
        if target_device.kind == "cuda":
            from underhfs.native import status

            native = status()
            if not native.available:
                raise RuntimeError(
                    "cannot move tensor to CUDA because the native core is unavailable: "
                    f"{native.reason}"
                )
            if not native.cuda_enabled:
                raise RuntimeError(
                    "cannot move tensor to CUDA because underHFS was built without CUDA support. "
                    "Rebuild with UNDERHFS_WITH_CUDA=ON after installing CUDA Toolkit."
                )
        out = Tensor(
            self.tolist(),
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype if dtype is None else DType(dtype),
            device=target_device,
            layout=self.layout if layout is None else Layout(layout),
        )
        if target_device.kind == "cuda":
            out._attach_cuda_storage()
        return out

    def cpu(self) -> Tensor:
        return self.to("cpu")

    def cuda(self, index: int = 0) -> Tensor:
        return self.to(Device("cuda", index))

    def zero_grad(self) -> None:
        self.grad = None

    def tolist(self) -> Any:
        if self.shape == ():
            return self._value_at(())

        def build(prefix: tuple[int, ...], dims: tuple[int, ...]) -> Any:
            if not dims:
                return self._value_at(prefix)
            return [build((*prefix, i), dims[1:]) for i in range(dims[0])]

        return build((), self.shape)

    def item(self) -> float:
        if self.numel() != 1:
            raise ValueError("item() is only valid for single-value tensors")
        return self._value_at(()) if self.shape == () else self._flat_values()[0]

    def argmax(self) -> int:
        if self.numel() == 0:
            raise ValueError("argmax() is not valid for empty tensors")
        best_index = 0
        values = self._flat_values()
        best_value = values[0]
        for index, value in enumerate(values[1:], start=1):
            if value > best_value:
                best_index = index
                best_value = value
        return best_index

    def _offset(self, index: tuple[int, ...]) -> int:
        if len(index) != len(self.shape):
            raise IndexError(f"expected {len(self.shape)} indices, got {len(index)}")
        return self._storage_offset + sum(i * s for i, s in zip(index, self.strides, strict=True))

    def _value_at(self, index: tuple[int, ...]) -> float:
        return self._storage[self._offset(index)]

    def _set_at(self, index: tuple[int, ...], value: float) -> None:
        self._storage[self._offset(index)] = value

    def _add_at(self, index: tuple[int, ...], value: float) -> None:
        self._storage[self._offset(index)] += value

    def _view(
        self,
        *,
        shape: Sequence[int],
        strides: Sequence[int],
        storage_offset: int,
        op: str,
        index_map: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
    ) -> Tensor:
        out = Tensor(
            [0.0] * _prod(shape),
            shape=shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op=op,
            _storage=self._storage,
            _storage_offset=storage_offset,
            _strides=strides,
            _version_ref=self._version_ref,
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                if index_map is None:
                    self._accumulate_grad(out.grad.reshape(self.shape))
                    return
                grad = zeros(self.shape, dtype=self.dtype, device=self.device, layout=self.layout)
                for out_index in _iter_indices(out.shape):
                    grad._add_at(index_map(out_index), out.grad._value_at(out_index))
                self._accumulate_grad(grad)

        out._backward = backward
        return out

    def _ensure_tensor(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return tensor(other, dtype=self.dtype, device=self.device, layout=self.layout)

    def _native_cpu_eligible(self) -> bool:
        return (
            self.device.kind == "cpu"
            and self.layout is Layout.DENSE
            and self.dtype is DType.FP32
            and self.is_contiguous()
            and self._storage_offset == 0
            and self.numel() == len(self._storage)
        )

    def _native_cuda_eligible(self) -> bool:
        return (
            self.device.kind == "cuda"
            and self.layout is Layout.DENSE
            and self.dtype in {DType.FP32, DType.FP16, DType.BF16}
            and self.is_contiguous()
            and self._storage_offset == 0
            and self.numel() == len(self._storage)
        )

    def _native_cuda_matmul_eligible(self) -> bool:
        return self._native_cuda_eligible() and self.dtype is DType.FP32

    def _attach_cuda_storage(self) -> None:
        if not self._native_cuda_eligible():
            return
        from underhfs.native import require_native

        core = require_native()
        if not bool(getattr(core, "cuda_enabled", False)):
            raise RuntimeError("CUDA Tensor storage is unavailable in this underHFS native build")
        storage_class = {
            DType.FP32: "CudaTensorF32",
            DType.FP16: "CudaTensorF16",
            DType.BF16: "CudaTensorBF16",
        }[self.dtype]
        if not hasattr(core, storage_class):
            raise RuntimeError(f"{self.dtype.value} CUDA Tensor storage is unavailable in this underHFS native build")
        self._native_cuda = getattr(core, storage_class)([float(value) for value in self._storage], list(self.shape))
        self.backend = "native_cuda"

    def _sync_from_cuda(self) -> None:
        if self._native_cuda is not None:
            self._storage = [float(value) for value in self._native_cuda.to_host()]
            self._storage_offset = 0
            self.strides = _contiguous_strides(self.shape)

    def _to_native_core(self):
        from underhfs.native import require_native

        core = require_native()
        return core.TensorCore(list(self._storage), list(self.shape))

    @classmethod
    def _from_native_core(
        cls,
        native_tensor,
        *,
        requires_grad: bool = False,
        children: Iterable[Tensor] = (),
        op: str = "",
    ) -> Tensor:
        out = cls(
            list(native_tensor.storage),
            shape=tuple(native_tensor.shape),
            requires_grad=requires_grad,
            _children=children,
            _op=op,
        )
        out.backend = "native_cpu"
        return out

    def _try_native_binary(self, rhs: Tensor, op: str) -> Tensor | None:
        if op not in {"add", "mul"}:
            return None
        if (
            self.shape == rhs.shape
            and self.dtype == rhs.dtype
            and self._native_cuda_eligible()
            and rhs._native_cuda_eligible()
        ):
            try:
                if self._native_cuda is None:
                    self._attach_cuda_storage()
                if rhs._native_cuda is None:
                    rhs._attach_cuda_storage()
                out = Tensor(
                    [0.0] * self.numel(),
                    shape=self.shape,
                    requires_grad=self.requires_grad or rhs.requires_grad,
                    dtype=self.dtype,
                    device=self.device,
                    layout=self.layout,
                    _children=(self, rhs),
                    _op=op,
                )
                if op == "add":
                    out._native_cuda = self._native_cuda.add(rhs._native_cuda)
                else:
                    out._native_cuda = self._native_cuda.mul(rhs._native_cuda)
                out.backend = "native_cuda"
                out._sync_from_cuda()
                return out
            except Exception:
                return None
        if self.shape != rhs.shape or not self._native_cpu_eligible() or not rhs._native_cpu_eligible():
            return None
        try:
            left_native = self._to_native_core()
            right_native = rhs._to_native_core()
            native_out = left_native.add(right_native) if op == "add" else left_native.mul(right_native)
        except Exception:
            return None
        return Tensor._from_native_core(
            native_out,
            requires_grad=self.requires_grad or rhs.requires_grad,
            children=(self, rhs),
            op=op,
        )

    def _check_compatible(self, other: Tensor, op: str) -> None:
        if self.device != other.device:
            raise RuntimeError(
                f"{op} received tensors on different devices: {self.device} and {other.device}. "
                "Move tensors explicitly with .to(...), .cpu(), or .cuda()."
            )
        if self.layout != other.layout:
            raise RuntimeError(
                f"{op} received tensors with different layouts: {self.layout.value} and "
                f"{other.layout.value}. Convert layout explicitly before the operation."
            )
        if self.dtype != other.dtype and self.shape != () and other.shape != ():
            raise RuntimeError(
                f"{op} received tensors with different dtypes: {self.dtype.value} and "
                f"{other.dtype.value}. Cast explicitly with .to(dtype=...)."
            )

    def _check_saved_versions(self) -> None:
        for child, expected in self._saved_versions.items():
            if child.version != expected:
                raise RuntimeError(
                    "autograd detected an in-place mutation before backward: "
                    f"op={self._op or 'leaf'} expected input version {expected}, "
                    f"but found {child.version}. Use out-of-place ops, clone(), or detach() "
                    "before mutating tensors needed for gradients."
                )

    def _accumulate_grad(self, grad: Tensor) -> None:
        if self.grad is None:
            self.grad = grad.detach()
            return
        self.grad = self.grad + grad

    def _binary(self, other: Any, op: str, fn: Callable[[float, float], float]) -> Tensor:
        rhs = self._ensure_tensor(other)
        self._check_compatible(rhs, op)
        shape = _broadcast_shape(self.shape, rhs.shape)
        out = self._try_native_binary(rhs, op)
        if out is None:
            values = [
                fn(
                    self._value_at(_broadcast_index(idx, self.shape)),
                    rhs._value_at(_broadcast_index(idx, rhs.shape)),
                )
                for idx in _iter_indices(shape)
            ]
            out = Tensor(
                values,
                shape=shape,
                requires_grad=self.requires_grad or rhs.requires_grad,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
                _children=(self, rhs),
                _op=op,
            )

        def backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if op == "add":
                    self._accumulate_grad(_unbroadcast(out.grad, self.shape))
                elif op == "sub":
                    self._accumulate_grad(_unbroadcast(out.grad, self.shape))
                elif op == "mul":
                    self._accumulate_grad(_unbroadcast(out.grad * rhs, self.shape))
                elif op == "div":
                    self._accumulate_grad(_unbroadcast(out.grad / rhs, self.shape))
            if rhs.requires_grad:
                if op == "add":
                    rhs._accumulate_grad(_unbroadcast(out.grad, rhs.shape))
                elif op == "sub":
                    rhs._accumulate_grad(_unbroadcast(-out.grad, rhs.shape))
                elif op == "mul":
                    rhs._accumulate_grad(_unbroadcast(out.grad * self, rhs.shape))
                elif op == "div":
                    rhs._accumulate_grad(_unbroadcast(-(out.grad * self) / (rhs * rhs), rhs.shape))

        out._backward = backward
        return out

    def __add__(self, other: Any) -> Tensor:
        return self._binary(other, "add", lambda a, b: a + b)

    def __radd__(self, other: Any) -> Tensor:
        return self + other

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(other, "sub", lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Tensor:
        return tensor(other, dtype=self.dtype, device=self.device, layout=self.layout) - self

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(other, "mul", lambda a, b: a * b)

    def __rmul__(self, other: Any) -> Tensor:
        return self * other

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(other, "div", lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> Tensor:
        return tensor(other, dtype=self.dtype, device=self.device, layout=self.layout) / self

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __pow__(self, power: float) -> Tensor:
        values = [value**power for value in self._flat_values()]
        out = Tensor(
            values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="pow",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(out.grad * (power * (self ** (power - 1))))

        out._backward = backward
        return out

    def __matmul__(self, other: Any) -> Tensor:
        rhs = self._ensure_tensor(other)
        self._check_compatible(rhs, "matmul")
        if self.ndim != 2 or rhs.ndim != 2:
            raise ValueError("matmul fallback currently supports 2D tensors")
        m, k = self.shape
        k2, n = rhs.shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch: {self.shape} @ {rhs.shape}")
        out = None
        if self._native_cuda_matmul_eligible() and rhs._native_cuda_matmul_eligible():
            try:
                if self._native_cuda is None:
                    self._attach_cuda_storage()
                if rhs._native_cuda is None:
                    rhs._attach_cuda_storage()
                out = Tensor(
                    [0.0] * (m * n),
                    shape=(m, n),
                    requires_grad=self.requires_grad or rhs.requires_grad,
                    dtype=self.dtype,
                    device=self.device,
                    layout=self.layout,
                    _children=(self, rhs),
                    _op="matmul",
                )
                out._native_cuda = self._native_cuda.matmul(rhs._native_cuda)
                out.backend = "native_cuda"
                out._sync_from_cuda()
            except Exception:
                out = None
        if self._native_cpu_eligible() and rhs._native_cpu_eligible():
            try:
                out = Tensor._from_native_core(
                    self._to_native_core().matmul(rhs._to_native_core()),
                    requires_grad=self.requires_grad or rhs.requires_grad,
                    children=(self, rhs),
                    op="matmul",
                )
            except Exception:
                out = None
        if out is None:
            values = []
            for i in range(m):
                for j in range(n):
                    values.append(sum(self._value_at((i, p)) * rhs._value_at((p, j)) for p in range(k)))
            out = Tensor(
                values,
                shape=(m, n),
                requires_grad=self.requires_grad or rhs.requires_grad,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
                _children=(self, rhs),
                _op="matmul",
            )

        def backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad @ rhs.transpose())
            if rhs.requires_grad:
                rhs._accumulate_grad(self.transpose() @ out.grad)

        out._backward = backward
        return out

    def add_(self, other: Any) -> Tensor:
        result = self + other
        if result.shape != self.shape:
            raise RuntimeError(f"in-place add cannot change tensor shape from {self.shape} to {result.shape}")
        for index in _iter_indices(self.shape):
            self._set_at(index, result._value_at(index))
        if self.is_contiguous() and self._storage_offset == 0 and self.numel() == len(self._storage):
            self._native_cuda = result._native_cuda
            self.backend = result.backend
        else:
            self._native_cuda = None
            self.backend = "python"
        self._version_ref[0] += 1
        return self

    def sum(self) -> Tensor:
        out = None
        if self._native_cuda_matmul_eligible():
            try:
                if self._native_cuda is None:
                    self._attach_cuda_storage()
                out = Tensor(
                    0.0,
                    requires_grad=self.requires_grad,
                    dtype=self.dtype,
                    device=self.device,
                    layout=self.layout,
                    _children=(self,),
                    _op="sum",
                )
                out._native_cuda = self._native_cuda.sum()
                out.backend = "native_cuda"
                out._sync_from_cuda()
            except Exception:
                out = None
        if self._native_cpu_eligible():
            try:
                out = Tensor._from_native_core(
                    self._to_native_core().sum(),
                    requires_grad=self.requires_grad,
                    children=(self,),
                    op="sum",
                )
            except Exception:
                out = None
        if out is None:
            out = Tensor(
                sum(self._flat_values()),
                requires_grad=self.requires_grad,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
                _children=(self,),
                _op="sum",
            )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(
                    ones(self.shape, dtype=self.dtype, device=self.device, layout=self.layout) * out.grad
                )

        out._backward = backward
        return out

    def mean(self) -> Tensor:
        return self.sum() / max(1, self.numel())

    def relu(self) -> Tensor:
        values = [max(0.0, value) for value in self._flat_values()]
        out = Tensor(
            values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="relu",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                mask = Tensor(
                    [1.0 if value > 0 else 0.0 for value in self._flat_values()],
                    shape=self.shape,
                    dtype=self.dtype,
                    device=self.device,
                    layout=self.layout,
                )
                self._accumulate_grad(out.grad * mask)

        out._backward = backward
        return out

    def tanh(self) -> Tensor:
        values = [tanh(value) for value in self._flat_values()]
        out = Tensor(
            values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="tanh",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                grad = Tensor(
                    [1.0 - value * value for value in out._storage],
                    shape=out.shape,
                    dtype=self.dtype,
                    device=self.device,
                    layout=self.layout,
                )
                self._accumulate_grad(out.grad * grad)

        out._backward = backward
        return out

    def exp(self) -> Tensor:
        values = [exp(value) for value in self._flat_values()]
        out = Tensor(
            values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="exp",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(out.grad * out)

        out._backward = backward
        return out

    def log(self) -> Tensor:
        values = [log(value) for value in self._flat_values()]
        out = Tensor(
            values,
            shape=self.shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="log",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(out.grad / self)

        out._backward = backward
        return out

    def sqrt(self) -> Tensor:
        return self ** 0.5

    def transpose(self) -> Tensor:
        if self.ndim != 2:
            raise ValueError("transpose fallback currently supports 2D tensors")
        rows, cols = self.shape
        values = [self._value_at((i, j)) for j in range(cols) for i in range(rows)]
        out = Tensor(
            values,
            shape=(cols, rows),
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="transpose",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(out.grad.transpose())

        out._backward = backward
        return out

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def view(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        actual_shape = _resolve_shape(shape, self.numel())
        if not self.is_contiguous():
            raise ValueError("view requires a contiguous tensor; call reshape() for a copied fallback")
        if _prod(actual_shape) != self.numel():
            raise ValueError(f"cannot view {self.shape} as {actual_shape}")
        out = self._view(
            shape=actual_shape,
            strides=_contiguous_strides(actual_shape),
            storage_offset=self._storage_offset,
            op="view",
        )
        return out

    def flatten(self) -> Tensor:
        return self.view(self.numel())

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        shape = _resolve_shape(shape, self.numel())
        if self.is_contiguous():
            return self.view(*shape)
        if _prod(shape) != self.numel():
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        out = Tensor(
            self._flat_values(),
            shape=shape,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
            _children=(self,),
            _op="reshape",
        )

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out._backward = backward
        return out

    def __getitem__(self, key: Any) -> Tensor:
        if not isinstance(key, tuple):
            key = (key,)
        if any(item is Ellipsis for item in key):
            raise IndexError("ellipsis indexing is not supported yet")
        if len(key) > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")
        key = (*key, *(slice(None) for _ in range(self.ndim - len(key))))
        storage_offset = self._storage_offset
        shape: list[int] = []
        strides: list[int] = []
        index_specs: list[int | range] = []
        for axis, item in enumerate(key):
            dim = self.shape[axis]
            stride = self.strides[axis]
            if isinstance(item, int):
                actual = item + dim if item < 0 else item
                if actual < 0 or actual >= dim:
                    raise IndexError(f"index {item} is out of bounds for axis {axis} with size {dim}")
                storage_offset += actual * stride
                index_specs.append(actual)
            elif isinstance(item, slice):
                start, stop, step = item.indices(dim)
                values = range(start, stop, step)
                length = len(values)
                shape.append(length)
                strides.append(stride * step)
                if length:
                    storage_offset += start * stride
                index_specs.append(values)
            else:
                raise IndexError("tensor indices must be integers or slices")

        def index_map(out_index: tuple[int, ...]) -> tuple[int, ...]:
            out_axis = 0
            parent: list[int] = []
            for spec in index_specs:
                if isinstance(spec, int):
                    parent.append(spec)
                else:
                    parent.append(spec[out_index[out_axis]])
                    out_axis += 1
            return tuple(parent)

        return self._view(
            shape=tuple(shape),
            strides=tuple(strides),
            storage_offset=storage_offset,
            op="slice",
            index_map=index_map,
        )

    def softmax(self) -> Tensor:
        if self.ndim == 1:
            row = self._flat_values()
            max_value = max(row)
            exps = [exp(v - max_value) for v in row]
            denom = sum(exps)
            out = Tensor(
                [v / denom for v in exps],
                shape=self.shape,
                requires_grad=self.requires_grad,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
                _children=(self,),
                _op="softmax",
            )
        elif self.ndim == 2:
            rows, cols = self.shape
            values = []
            for i in range(rows):
                row = [self._value_at((i, j)) for j in range(cols)]
                max_value = max(row)
                exps = [exp(v - max_value) for v in row]
                denom = sum(exps)
                values.extend(v / denom for v in exps)
            out = Tensor(
                values,
                shape=self.shape,
                requires_grad=self.requires_grad,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
                _children=(self,),
                _op="softmax",
            )
        else:
            raise ValueError("softmax fallback supports 1D or 2D tensors")

        def backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = zeros(self.shape, dtype=self.dtype, device=self.device, layout=self.layout)
            if self.ndim == 1:
                dot = sum(out.grad._value_at((j,)) * out._value_at((j,)) for j in range(self.shape[0]))
                for i in range(self.shape[0]):
                    grad._set_at((i,), out._value_at((i,)) * (out.grad._value_at((i,)) - dot))
            else:
                rows, cols = self.shape
                for row in range(rows):
                    dot = sum(
                        out.grad._value_at((row, j)) * out._value_at((row, j))
                        for j in range(cols)
                    )
                    for col in range(cols):
                        grad._set_at(
                            (row, col),
                            out._value_at((row, col)) * (out.grad._value_at((row, col)) - dot),
                        )
            self._accumulate_grad(grad)

        out._backward = backward
        return out

    def backward(self, grad: Tensor | None = None) -> None:
        if grad is None:
            if self.numel() != 1:
                raise ValueError("grad must be provided for non-scalar tensors")
            grad = tensor(1.0, dtype=self.dtype, device=self.device, layout=self.layout)
        self.grad = grad
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            node._check_saved_versions()
            node._backward()

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.tolist()!r}, shape={self.shape}, dtype={self.dtype.value}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )


def tensor(data: Any, **kwargs: Any) -> Tensor:
    return Tensor(data, **kwargs)


def zeros(shape: int | Sequence[int], **kwargs: Any) -> Tensor:
    actual = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor([0.0] * _prod(actual), shape=actual, **kwargs)


def ones(shape: int | Sequence[int], **kwargs: Any) -> Tensor:
    actual = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor([1.0] * _prod(actual), shape=actual, **kwargs)


def arange(stop: int, **kwargs: Any) -> Tensor:
    return Tensor([float(i) for i in range(stop)], shape=(stop,), **kwargs)


def kaiming_uniform(rows: int, cols: int) -> Tensor:
    bound = sqrt(6.0 / max(1, rows + cols))
    values = [(((i * 1103515245 + 12345) % 10000) / 5000.0 - 1.0) * bound for i in range(rows * cols)]
    return Tensor(values, shape=(rows, cols), requires_grad=True)


def uniform(shape: Sequence[int], bound: float, *, requires_grad: bool = False) -> Tensor:
    total = _prod(shape)
    values = [
        (((i * 1103515245 + 12345) % 10000) / 5000.0 - 1.0) * bound
        for i in range(total)
    ]
    return Tensor(values, shape=tuple(shape), requires_grad=requires_grad)
