from __future__ import annotations

from collections import OrderedDict
from math import sqrt
from typing import Iterable, Iterator

from underhfs import functional as F
from underhfs.tensor import Tensor, arange, kaiming_uniform, ones, tensor, uniform, zeros


class Parameter(Tensor):
    def __init__(self, data, **kwargs) -> None:
        if isinstance(data, Tensor):
            super().__init__(data.tolist(), shape=data.shape, requires_grad=True, dtype=data.dtype, device=data.device, layout=data.layout)
        else:
            super().__init__(data, requires_grad=True, **kwargs)


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        yield from self._parameters.values()
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, parameter in self._parameters.items():
            yield f"{prefix}{name}", parameter
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{name}.")

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, list | float]:
        return {name: parameter.detach().tolist() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, list | float]) -> None:
        for name, parameter in self.named_parameters():
            if name not in state:
                raise KeyError(f"missing parameter {name}")
            replacement = tensor(state[name])
            parameter._storage = replacement._storage
            parameter.shape = replacement.shape
            parameter.strides = replacement.strides
            parameter._version += 1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(kaiming_uniform(out_features, in_features))
        self.bias = Parameter(zeros((out_features,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = list(modules)
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()) -> None:
        super().__init__()
        self.modules = list(modules)
        for index, module in enumerate(self.modules):
            setattr(self, str(index), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.modules)

    def append(self, module: Module) -> None:
        self.modules.append(module)
        setattr(self, str(len(self.modules) - 1), module)


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Parameter(ones((features,)))
        self.bias = Parameter(zeros((features,)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("LayerNorm fallback supports [batch, features]")
        rows, cols = x.shape
        values = []
        for i in range(rows):
            row = [x._value_at((i, j)) for j in range(cols)]
            mean = sum(row) / cols
            var = sum((v - mean) ** 2 for v in row) / cols
            denom = (var + self.eps) ** 0.5
            values.extend((row[j] - mean) / denom for j in range(cols))
        return tensor(values, shape=x.shape, requires_grad=x.requires_grad) * self.weight + self.bias


class RMSNorm(Module):
    def __init__(self, features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = Parameter(ones((features,)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("RMSNorm fallback supports [batch, features]")
        rows, cols = x.shape
        values = []
        for i in range(rows):
            row = [x._value_at((i, j)) for j in range(cols)]
            rms = (sum(v * v for v in row) / cols + self.eps) ** 0.5
            values.extend(v / rms for v in row)
        return tensor(values, shape=x.shape, requires_grad=x.requires_grad) * self.weight


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = Parameter(kaiming_uniform(num_embeddings, embedding_dim))

    def forward(self, indices: Tensor) -> Tensor:
        if indices.ndim != 1:
            raise ValueError("Embedding fallback supports 1D indices")
        values = []
        for raw_index in indices._storage:
            row = int(raw_index)
            if row < 0 or row >= self.weight.shape[0]:
                raise ValueError(f"embedding index {row} outside [0, {self.weight.shape[0]})")
            values.extend(self.weight._value_at((row, j)) for j in range(self.weight.shape[1]))
        out = tensor(
            values,
            shape=(indices.numel(), self.weight.shape[1]),
            requires_grad=self.weight.requires_grad,
        )
        out._prev = {self.weight}
        out._op = "embedding"

        def backward() -> None:
            if out.grad is None or not self.weight.requires_grad:
                return
            grad = zeros(self.weight.shape)
            for position, raw_index in enumerate(indices._storage):
                row = int(raw_index)
                for col in range(self.weight.shape[1]):
                    grad._add_at((row, col), out.grad._value_at((position, col)))
            self.weight._accumulate_grad(grad)

        out._backward = backward
        return out


class MLP(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = Sequential(Linear(in_features, hidden_features), GELU(), Linear(hidden_features, out_features))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SelfAttention(Module):
    def __init__(self, features: int, bias: bool = True) -> None:
        super().__init__()
        self.features = features
        self.q_proj = Linear(features, features, bias=bias)
        self.k_proj = Linear(features, features, bias=bias)
        self.v_proj = Linear(features, features, bias=bias)
        self.out_proj = Linear(features, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("SelfAttention fallback expects [tokens, features]")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (q @ k.T) / sqrt(self.features)
        weights = scores.softmax()
        return self.out_proj(weights @ v)


class CausalSelfAttention(SelfAttention):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("CausalSelfAttention fallback expects [tokens, features]")
        tokens = x.shape[0]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (q @ k.T) / sqrt(self.features)
        mask_values = [
            0.0 if col <= row else -1.0e9
            for row in range(tokens)
            for col in range(tokens)
        ]
        scores = scores + tensor(mask_values, shape=(tokens, tokens))
        weights = scores.softmax()
        return self.out_proj(weights @ v)


class TransformerBlock(Module):
    def __init__(self, features: int, hidden_features: int, *, causal: bool = False) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(features)
        self.attn = CausalSelfAttention(features) if causal else SelfAttention(features)
        self.mlp_norm = RMSNorm(features)
        self.mlp = MLP(features, hidden_features, features)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        features: int,
        hidden_features: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, features)
        self.position_embedding = Embedding(max_seq_len, features)
        self.blocks = ModuleList(
            TransformerBlock(features, hidden_features, causal=True) for _ in range(layers)
        )
        self.norm = RMSNorm(features)
        self.head = Linear(features, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 1:
            raise ValueError("TransformerLM fallback expects 1D token ids")
        seq_len = token_ids.shape[0]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        x = self.token_embedding(token_ids) + self.position_embedding(arange(seq_len))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def generate(self, token_ids: Tensor, max_new_tokens: int) -> Tensor:
        if token_ids.ndim != 1:
            raise ValueError("TransformerLM.generate fallback expects 1D token ids")
        generated = [int(value) for value in token_ids._storage]
        for _ in range(max_new_tokens):
            if len(generated) >= self.max_seq_len:
                context = generated[-self.max_seq_len :]
            else:
                context = generated
            logits = self(tensor(context))
            last_row = [
                logits._value_at((logits.shape[0] - 1, col))
                for col in range(logits.shape[1])
            ]
            next_token = tensor(last_row).argmax()
            generated.append(next_token)
        return tensor(generated)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight = Parameter(
            uniform(
                (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
                (1.0 / max(1, fan_in)) ** 0.5,
            )
        )
        self.bias = Parameter(zeros((out_channels,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Conv2d fallback expects input shape [batch, channels, height, width]")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("kernel/stride/padding produce empty output")

        values: list[float] = []
        for n in range(batch):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        acc = self.bias._value_at((oc,)) if self.bias is not None else 0.0
                        for ic in range(self.in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    ih = oh * stride_h + kh - pad_h
                                    iw = ow * stride_w + kw - pad_w
                                    if 0 <= ih < height and 0 <= iw < width:
                                        acc += x._value_at((n, ic, ih, iw)) * self.weight._value_at((oc, ic, kh, kw))
                        values.append(acc)

        out = tensor(
            values,
            shape=(batch, self.out_channels, out_h, out_w),
            requires_grad=x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad),
        )
        parents = {x, self.weight}
        if self.bias is not None:
            parents.add(self.bias)
        out._prev = parents
        out._op = "conv2d"

        def backward() -> None:
            if out.grad is None:
                return
            grad_x = zeros(x.shape) if x.requires_grad else None
            grad_w = zeros(self.weight.shape) if self.weight.requires_grad else None
            grad_b = zeros(self.bias.shape) if self.bias is not None and self.bias.requires_grad else None
            for n in range(batch):
                for oc in range(self.out_channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            go = out.grad._value_at((n, oc, oh, ow))
                            if grad_b is not None:
                                grad_b._add_at((oc,), go)
                            for ic in range(self.in_channels):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        ih = oh * stride_h + kh - pad_h
                                        iw = ow * stride_w + kw - pad_w
                                        if 0 <= ih < height and 0 <= iw < width:
                                            if grad_x is not None:
                                                grad_x._add_at(
                                                    (n, ic, ih, iw),
                                                    self.weight._value_at((oc, ic, kh, kw)) * go,
                                                )
                                            if grad_w is not None:
                                                grad_w._add_at(
                                                    (oc, ic, kh, kw),
                                                    x._value_at((n, ic, ih, iw)) * go,
                                                )
            if grad_x is not None:
                x._accumulate_grad(grad_x)
            if grad_w is not None:
                self.weight._accumulate_grad(grad_w)
            if grad_b is not None and self.bias is not None:
                self.bias._accumulate_grad(grad_b)

        out._backward = backward
        return out


class PolicyValueHead(Module):
    def __init__(self, features: int, actions: int) -> None:
        super().__init__()
        self.policy = Linear(features, actions)
        self.value = Linear(features, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy(x), self.value(x)


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, self.reduction)


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target, self.reduction)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)
