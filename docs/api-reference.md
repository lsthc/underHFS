# underHFS API Reference

This reference is intentionally compact while the framework is still in its
native runtime bring-up phase.

## Tensor

- `underhfs.tensor(data, **kwargs)` creates a Tensor.
- `Tensor.to(device=None, dtype=None, layout=None)` casts metadata and moves to
  CUDA when the native backend is available.
- `Tensor.backward(grad=None)` runs eager reverse-mode autograd.
- `Tensor.view`, `reshape`, `flatten`, slicing, `sum`, `mean`, `softmax`,
  `relu`, `tanh`, `exp`, `log`, `matmul` are available in the portable runtime;
  supported dense fp32 CPU/CUDA paths use native fast paths when `_core` is built.

## Autograd

- `underhfs.autograd.backward(tensor, grad=None)`
- `underhfs.autograd.no_grad()`
- `underhfs.autograd.jvp(function, primals, tangents)`
- `underhfs.autograd.checkpoint(function, *args, **kwargs)`
- `underhfs.autograd.checkpoint_sequential(functions, segments, input)`

## Runtime

- `underhfs.cuda.capability_matrix()`
- `underhfs.cuda.require_kernel(op, device="cpu", dtype="fp32")`
- `underhfs.runtime.MemoryPlanner`
- `underhfs.runtime.OffloadExecutor`
- `OffloadExecutor.offload_tensor(tensor, tier=MemoryTier.NVME)`
- `OffloadExecutor.prefetch_tensor(handle)`
- `OffloadExecutor.load_tensor(handle, device=None)`
- `OffloadExecutor.release(handle)`
- `underhfs.runtime.NetworkOffloadServer(host="127.0.0.1", port=0)`
- `underhfs.runtime.NetworkOffloadClient(base_url)`
- `NetworkOffloadClient.offload_tensor(tensor)`
- `NetworkOffloadClient.load_tensor(handle, device=None)`
- `NetworkOffloadClient.release(handle)`

## Optimizers

- `underhfs.optim.SGD`
- `underhfs.optim.AdamW`
- `underhfs.optim.FusedAdamW`
- Native CUDA builds expose `_core.cuda_fused_adamw_f32(...)` for fp32 fused
  AdamW parameter/state updates.
- Native CUDA builds expose `_core.cuda_attention_f32(q, k, v, tokens, features, scale, causal)`
  for supported contiguous fp32 attention inference paths.
- cuDNN builds expose `_core.cudnn_conv2d_forward_f32(...)`,
  `_core.cudnn_conv2d_backward_input_f32(...)`, and
  `_core.cudnn_conv2d_backward_weight_f32(...)` for fp32 NCHW dense
  convolution paths.

## Distributed

- `underhfs.distributed.DistributedPolicy(world_size=1, rank=0, backend="nccl")`
- `underhfs.distributed.process_group(policy=None)`
- `underhfs.distributed.nccl_runtime_plan(policy)`
- `underhfs.distributed.DistributedDataParallel(module, policy=None)`
- NCCL native builds expose `_core.NcclProcessGroup(rank, world_size, unique_id_hex="")`
  and `_core.nccl_create_unique_id_hex()`.

## Compile

- `underhfs.compile.compile(function=None, policy=None)`
- `underhfs.compile.explain(function, *args, policy=None, **kwargs)`
- `CompileReport.to_dict()` includes graph, guards, fusion groups, cache stats,
  and an eager fused execution plan.
- `CompiledKernel.dispatch(*inputs, op=None, scale=None, causal=False)` runs
  supported native CUDA attention or fused add/mul/sum dispatch paths when the
  lowered backend is executable.

## Serving

- `underhfs.serve.serve(handler)`
- `underhfs.serve.serve_http(handler, config=None)`
- `underhfs.serve.serve_websocket(handler, config=None)`
- `underhfs.serve.serve_websocket_loop(handler, config=None)`
- `underhfs.serve.serve_grpc(handler, config=None)`
- `underhfs.serve.serve_cpp(executable=None)`
- `underhfs.serve.serve_grpc_manifest(config=None)`
- `underhfs.serve.serve_cpp_manifest(config=None)`
- `underhfs.serve.open_stream(source, kind=StreamSourceKind.FILE)`

`open_stream` supports built-in file byte streaming, OpenCV-backed webcam/file
frames when `cv2` is installed, and FFmpeg subprocess streaming for RTSP/HLS
when `ffmpeg` is available on `PATH`.

## Serialization

- `underhfs.serialization.save_checkpoint(path, state=..., metadata=...)`
- `underhfs.serialization.save_binary_state_dict(path, state)`
- `underhfs.serialization.export_onnx(path, model_name=..., state=..., inputs=...)`
- `underhfs.serialization.import_onnx(path)`
- `underhfs.serialization.load_onnx_state_dict(path)`
