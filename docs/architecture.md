# underHFS Architecture

underHFS is split into four layers:

1. Python API: PyTorch-style ergonomics, modules, optimizers, policies, CLI, and
   serving surfaces.
2. Native core: Tensor storage, dtype/device/layout metadata, allocator, eager
   execution, autograd metadata, and pybind11 bindings.
3. CUDA/CPU kernels: vendor-backed fast paths for cuBLAS/cuDNN/NCCL plus custom
   kernels and CPU kernels.
4. Runtime/compiler: graph IR, partial dynamic shapes, fusion, stream-aware
   scheduling, hierarchical memory, offload, recompute, and distributed policies.

The current Python fallback is intentionally small but executable. It anchors
the public semantics while the C++/CUDA backend comes online.
