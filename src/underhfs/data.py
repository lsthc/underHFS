from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Generic, Iterator, Protocol, Sequence, TypeVar

T = TypeVar("T")


class Dataset(Protocol[T]):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T: ...


@dataclass
class TensorDataset(Generic[T]):
    tensors: Sequence[T]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> T:
        return self.tensors[index]


class DataLoader(Generic[T]):
    def __init__(self, dataset: Dataset[T], batch_size: int = 1, shuffle: bool = False, seed: int = 0) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[list[T]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            Random(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield [self.dataset[index] for index in indices[start : start + self.batch_size]]
