from __future__ import annotations

from dataclasses import dataclass

from underhfs import tensor
from underhfs.tensor import Tensor


@dataclass(frozen=True)
class ByteTokenizer:
    """A tiny byte-level tokenizer for local smoke tests and bootstrap models."""

    vocab_size: int = 256

    def encode(self, text: str) -> Tensor:
        return tensor(list(text.encode("utf-8")))

    def decode(self, tokens: Tensor) -> str:
        values = bytes(max(0, min(255, int(value))) for value in tokens._storage)
        return values.decode("utf-8", errors="ignore")
