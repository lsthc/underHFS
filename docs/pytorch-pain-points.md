# PyTorch Pain Points underHFS Intentionally Guards Against

This note records early design guardrails for underHFS before the native
C++/CUDA backend lands.

## Pain points observed in the PyTorch ecosystem

- CUDA out-of-memory messages often involve allocated vs reserved memory,
  fragmentation, and unclear recovery paths.
- In-place tensor mutation can invalidate autograd graphs and fail late during
  backward.
- Reproducibility requires multiple knobs around deterministic algorithms,
  random seeds, cuDNN benchmarking, and DataLoader worker seeding.
- CUDA multiprocessing has accelerator-specific pitfalls such as poison fork.
- Device, dtype, and layout mismatches often appear inside larger model code,
  far from the original mistake.
- Installation and build environments are fragile when the NVIDIA driver is
  present but CUDA Toolkit, `nvcc`, compiler tools, or Ninja are missing.

## underHFS guardrails now in place

- Autograd records input tensor versions and raises a clear error if a tensor
  needed for backward was mutated in-place.
- Binary tensor ops fail explicitly on device/layout mismatch and on non-scalar
  dtype mismatch instead of silently guessing.
- `underhfs doctor` reports Python, CUDA visibility, native core status, and
  required tool availability.
- The memory planner makes tier placement explicit across VRAM/RAM/NVMe instead
  of pretending every model fits in one device.
- CUDA movement fails loudly while the native backend is unavailable.

## References

- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness
- PyTorch deterministic algorithms API: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- PyTorch multiprocessing notes: https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
- PyTorch issue about CUDA OOM recovery: https://github.com/pytorch/pytorch/issues/27600
