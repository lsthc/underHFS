from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

from underhfs.cuda import MemoryPolicy, MemoryTier, memory_budgets
from underhfs.tensor import DType, Tensor, tensor


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


@dataclass(frozen=True)
class OffloadHandle:
    id: str
    tier: MemoryTier
    path: str
    bytes: int
    shape: tuple[int, ...]
    dtype: DType
    device: str

    def to_dict(self) -> dict[str, str | int | list[int]]:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "path": self.path,
            "bytes": self.bytes,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "device": self.device,
        }


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


class OffloadExecutor:
    def __init__(self, policy: MemoryPolicy, root: str | Path | None = None) -> None:
        self.policy = policy
        scratch = root or policy.scratch_path or ".underhfs-offload"
        self.root = Path(scratch)
        self.root.mkdir(parents=True, exist_ok=True)

    def offload_tensor(self, value: Tensor, tier: MemoryTier = MemoryTier.NVME) -> OffloadHandle:
        if tier not in {MemoryTier.NVME, MemoryTier.NETWORK}:
            raise ValueError("offload executor currently supports NVMe/network tiers")
        if tier is MemoryTier.NETWORK:
            raise RuntimeError("network offload requires a configured remote transport")
        ident = uuid4().hex
        path = self.root / f"{ident}.uhfsoffload.json"
        payload = {
            "format": "underhfs.offload-tensor",
            "id": ident,
            "shape": list(value.shape),
            "dtype": value.dtype.value,
            "device": str(value.device),
            "data": value.detach().tolist(),
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        path.write_bytes(encoded)
        return OffloadHandle(
            id=ident,
            tier=tier,
            path=str(path),
            bytes=len(encoded),
            shape=value.shape,
            dtype=value.dtype,
            device=str(value.device),
        )

    def load_tensor(self, handle: OffloadHandle) -> Tensor:
        payload = json.loads(Path(handle.path).read_text(encoding="utf-8"))
        if payload.get("format") != "underhfs.offload-tensor":
            raise ValueError("not an underhfs offload tensor")
        if payload.get("id") != handle.id:
            raise ValueError("offload handle id mismatch")
        return tensor(payload["data"], dtype=handle.dtype)

    def release(self, handle: OffloadHandle) -> None:
        Path(handle.path).unlink(missing_ok=True)


def planner_from_system(policy: MemoryPolicy | None = None, *, vram_fraction: float = 0.9) -> MemoryPlanner:
    actual_policy = policy or MemoryPolicy()
    return MemoryPlanner(actual_policy, budgets=memory_budgets(vram_fraction=vram_fraction))
