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

## Native core contract

The native `_core` extension exposes `TensorCore` as the first C++ execution
object. Its initial contract mirrors the Python fallback for contiguous dense
CPU tensors:

- storage and shape validation
- contiguous stride derivation
- `add`, `mul`, `matmul`, and `sum`
- deterministic exceptions for unsupported shapes

`underhfs.native.probe()` exercises this contract when the extension is built.
`underhfs doctor` includes the probe output once `_core` is importable.

Native CPU availability is intentionally separate from CUDA availability.
The `_core.cuda_enabled` flag is false unless the extension was built with
`UNDERHFS_WITH_CUDA=ON`; `.cuda()` must fail clearly when only CPU native code is
present.
