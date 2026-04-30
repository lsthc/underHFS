from __future__ import annotations

from dataclasses import dataclass

from underhfs.cuda import MemoryPolicy, MemoryTier, memory_budgets
from underhfs.tensor import DType, Tensor


_DTYPE_BYTES = {
    DType.FP32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP8_E4M3: 1,
    DType.FP8_E5M2: 1,
    DType.INT8: 1,
    DType.INT4: 0.5,
}


@dataclass(frozen=True)
class Placement:
    tier: MemoryTier
    bytes: int
    reason: str


@dataclass
class TierBudget:
    tier: MemoryTier
    capacity_bytes: int
    used_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes)

    def reserve(self, size_bytes: int) -> bool:
        if size_bytes > self.available_bytes:
            return False
        self.used_bytes += size_bytes
        return True


class MemoryPlanner:
    def __init__(self, policy: MemoryPolicy, budgets: dict[MemoryTier, int]) -> None:
        self.policy = policy
        self.budgets = {
            tier: TierBudget(tier, budgets.get(tier, 0))
            for tier in policy.tiers
        }

    def tensor_size_bytes(self, tensor: Tensor) -> int:
        return int(tensor.numel() * _DTYPE_BYTES[tensor.dtype])

    def place_tensor(self, tensor: Tensor) -> Placement:
        return self.place_bytes(self.tensor_size_bytes(tensor))

    def place_bytes(self, size: int) -> Placement:
        if size < 0:
            raise ValueError("size must be non-negative")
        for tier in self.policy.tiers:
            budget = self.budgets[tier]
            if budget.reserve(size):
                return Placement(tier=tier, bytes=size, reason="fits-budget")
        if not self.policy.allow_offload:
            raise MemoryError(f"tensor requires {size} bytes, but no configured tier has capacity")
        final_tier = self.policy.tiers[-1]
        self.budgets[final_tier].used_bytes += size
        return Placement(tier=final_tier, bytes=size, reason="oversubscribed-offload")

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            tier.value: {
                "capacity_bytes": budget.capacity_bytes,
                "used_bytes": budget.used_bytes,
                "available_bytes": budget.available_bytes,
            }
            for tier, budget in self.budgets.items()
        }


def planner_from_system(policy: MemoryPolicy | None = None, *, vram_fraction: float = 0.9) -> MemoryPlanner:
    actual_policy = policy or MemoryPolicy()
    return MemoryPlanner(actual_policy, budgets=memory_budgets(vram_fraction=vram_fraction))
