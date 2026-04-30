from underhfs import tensor
from underhfs.cuda import MemoryPolicy, MemoryTier
from underhfs.runtime import MemoryPlanner
from underhfs.tensor import DType


def test_memory_planner_places_across_tiers():
    planner = MemoryPlanner(
        MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.RAM)),
        budgets={MemoryTier.VRAM: 8, MemoryTier.RAM: 64},
    )
    fp32 = tensor([1.0, 2.0], dtype=DType.FP32)
    larger = tensor([1.0, 2.0, 3.0], dtype=DType.FP32)
    first = planner.place_tensor(fp32)
    second = planner.place_tensor(larger)
    assert first.tier is MemoryTier.VRAM
    assert second.tier is MemoryTier.RAM
    assert planner.snapshot()["vram"]["used_bytes"] == 8


def test_memory_planner_offload_oversubscription():
    planner = MemoryPlanner(
        MemoryPolicy(tiers=(MemoryTier.VRAM, MemoryTier.NVME), allow_offload=True),
        budgets={MemoryTier.VRAM: 0, MemoryTier.NVME: 1},
    )
    placement = planner.place_tensor(tensor([1.0], dtype=DType.FP32))
    assert placement.tier is MemoryTier.NVME
    assert placement.reason == "oversubscribed-offload"
