#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef UNDERHFS_WITH_CUDNN
#include <cudnn.h>
#endif

#include <numeric>
#include <mutex>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels.hpp"

extern "C" __global__ void underhfs_add_f32(
    const float* left, const float* right, float* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = left[index] + right[index];
  }
}

extern "C" __global__ void underhfs_mul_f32(
    const float* left, const float* right, float* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = left[index] * right[index];
  }
}

extern "C" __global__ void underhfs_sum_blocks_f32(const float* values, float* partials, int n) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  float acc = 0.0f;
  if (index < n) {
    acc += values[index];
  }
  if (index + blockDim.x < n) {
    acc += values[index + blockDim.x];
  }
  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = scratch[0];
  }
}

extern "C" __global__ void underhfs_f32_to_f16(const float* src, __half* dst, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = __float2half(src[index]);
  }
}

extern "C" __global__ void underhfs_f16_to_f32(const __half* src, float* dst, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = __half2float(src[index]);
  }
}

extern "C" __global__ void underhfs_add_f16(
    const __half* left, const __half* right, __half* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = __hadd(left[index], right[index]);
  }
}

extern "C" __global__ void underhfs_mul_f16(
    const __half* left, const __half* right, __half* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = __hmul(left[index], right[index]);
  }
}

extern "C" __global__ void underhfs_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = __float2bfloat16(src[index]);
  }
}

extern "C" __global__ void underhfs_bf16_to_f32(const __nv_bfloat16* src, float* dst, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = __bfloat162float(src[index]);
  }
}

extern "C" __global__ void underhfs_add_bf16(
    const __nv_bfloat16* left, const __nv_bfloat16* right, __nv_bfloat16* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = __float2bfloat16(__bfloat162float(left[index]) + __bfloat162float(right[index]));
  }
}

extern "C" __global__ void underhfs_mul_bf16(
    const __nv_bfloat16* left, const __nv_bfloat16* right, __nv_bfloat16* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = __float2bfloat16(__bfloat162float(left[index]) * __bfloat162float(right[index]));
  }
}

extern "C" __global__ void underhfs_fused_adamw_f32(
    float* param,
    const float* grad,
    float* m,
    float* v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float decayed_grad = grad[index] + weight_decay * param[index];
    const float next_m = beta1 * m[index] + (1.0f - beta1) * decayed_grad;
    const float next_v = beta2 * v[index] + (1.0f - beta2) * decayed_grad * decayed_grad;
    const float m_hat = next_m / bias_correction1;
    const float v_hat = next_v / bias_correction2;
    param[index] -= lr * m_hat / (sqrtf(v_hat) + eps);
    m[index] = next_m;
    v[index] = next_v;
  }
}

extern "C" __global__ void underhfs_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int tokens,
    int features,
    float scale,
    int causal) {
  int row = blockIdx.x;
  int feature = threadIdx.x;
  if (row >= tokens || feature >= features) {
    return;
  }
  float max_score = -3.402823466e+38F;
  for (int col = 0; col < tokens; ++col) {
    if (causal && col > row) {
      continue;
    }
    float score = 0.0f;
    for (int f = 0; f < features; ++f) {
      score += q[row * features + f] * k[col * features + f];
    }
    score *= scale;
    max_score = fmaxf(max_score, score);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int col = 0; col < tokens; ++col) {
    if (causal && col > row) {
      continue;
    }
    float score = 0.0f;
    for (int f = 0; f < features; ++f) {
      score += q[row * features + f] * k[col * features + f];
    }
    const float weight = expf(score * scale - max_score);
    denom += weight;
    acc += weight * v[col * features + feature];
  }
  out[row * features + feature] = acc / denom;
}

namespace underhfs {

namespace {

void check_cuda(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
  }
}

void check_cublas(cublasStatus_t status, const char* context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

#ifdef UNDERHFS_WITH_CUDNN
void check_cudnn(cudnnStatus_t status, const char* context) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": " + cudnnGetErrorString(status));
  }
}
#endif

class CublasHandle {
 public:
  CublasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

cublasHandle_t cublas_handle() {
  thread_local CublasHandle handle;
  return handle.get();
}

class CudaRuntime {
 public:
  CudaRuntime() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~CudaRuntime() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaRuntime(const CudaRuntime&) = delete;
  CudaRuntime& operator=(const CudaRuntime&) = delete;

  cudaStream_t stream() const { return stream_; }

  void record_launch() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++launches_;
  }

  void record_copy() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++copies_;
  }

  void synchronize(const char* context) {
    check_cuda(cudaStreamSynchronize(stream_), context);
    std::lock_guard<std::mutex> lock(mutex_);
    ++synchronizations_;
  }

  std::unordered_map<std::string, std::size_t> stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        {"non_blocking_streams", 1},
        {"launches", launches_},
        {"copies", copies_},
        {"synchronizations", synchronizations_},
    };
  }

 private:
  cudaStream_t stream_ = nullptr;
  mutable std::mutex mutex_;
  std::size_t launches_ = 0;
  std::size_t copies_ = 0;
  std::size_t synchronizations_ = 0;
};

CudaRuntime& cuda_runtime() {
  static CudaRuntime runtime;
  return runtime;
}

class CudaCachingAllocator {
 public:
  float* allocate(std::size_t bytes, const char* context) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& bucket = free_blocks_[bytes];
    if (!bucket.empty()) {
      float* ptr = bucket.back();
      bucket.pop_back();
      cached_bytes_ -= bytes;
      active_bytes_ += bytes;
      ++reuses_;
      return ptr;
    }
    float* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), context);
    active_bytes_ += bytes;
    allocated_bytes_ += bytes;
    ++allocations_;
    return ptr;
  }

  void release(float* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    active_bytes_ -= bytes;
    cached_bytes_ += bytes;
    free_blocks_[bytes].push_back(ptr);
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, bucket] : free_blocks_) {
      for (float* ptr : bucket) {
        cudaFree(ptr);
      }
      bucket.clear();
    }
    cached_bytes_ = 0;
  }

  std::unordered_map<std::string, std::size_t> stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        {"active_bytes", active_bytes_},
        {"cached_bytes", cached_bytes_},
        {"allocated_bytes", allocated_bytes_},
        {"allocations", allocations_},
        {"reuses", reuses_},
    };
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::size_t, std::vector<float*>> free_blocks_;
  std::size_t active_bytes_ = 0;
  std::size_t cached_bytes_ = 0;
  std::size_t allocated_bytes_ = 0;
  std::size_t allocations_ = 0;
  std::size_t reuses_ = 0;
};

CudaCachingAllocator& cuda_allocator() {
  static CudaCachingAllocator allocator;
  return allocator;
}

}  // namespace

CudaTensorF32::CudaTensorF32(const std::vector<float>& host, std::vector<std::size_t> shape)
    : shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
  if (numel_ != host.size()) {
    throw std::invalid_argument("CudaTensorF32 host size does not match shape");
  }
  device_ = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorF32 cudaMalloc");
  try {
    check_cuda(cudaMemcpyAsync(device_, host.data(), numel_ * sizeof(float), cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "CudaTensorF32 host-to-device copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("CudaTensorF32 host-to-device sync");
  } catch (...) {
    cuda_allocator().release(device_, numel_ * sizeof(float));
    device_ = nullptr;
    throw;
  }
}

CudaTensorF32::CudaTensorF32(float* device, std::vector<std::size_t> shape)
    : device_(device), shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
}

CudaTensorF32::~CudaTensorF32() {
  if (device_ != nullptr) {
    cuda_allocator().release(device_, numel_ * sizeof(float));
  }
}

CudaTensorF32::CudaTensorF32(CudaTensorF32&& other) noexcept
    : device_(other.device_), shape_(std::move(other.shape_)), numel_(other.numel_) {
  other.device_ = nullptr;
  other.numel_ = 0;
}

CudaTensorF32& CudaTensorF32::operator=(CudaTensorF32&& other) noexcept {
  if (this != &other) {
    if (device_ != nullptr) {
      cuda_allocator().release(device_, numel_ * sizeof(float));
    }
    device_ = other.device_;
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    other.device_ = nullptr;
    other.numel_ = 0;
  }
  return *this;
}

const std::vector<std::size_t>& CudaTensorF32::shape() const { return shape_; }

std::size_t CudaTensorF32::numel() const { return numel_; }

std::vector<float> CudaTensorF32::to_host() const {
  std::vector<float> host(numel_, 0.0f);
  check_cuda(cudaMemcpyAsync(host.data(), device_, numel_ * sizeof(float), cudaMemcpyDeviceToHost,
                             cuda_runtime().stream()),
             "CudaTensorF32 device-to-host copy");
  cuda_runtime().record_copy();
  cuda_runtime().synchronize("CudaTensorF32 device-to-host sync");
  return host;
}

CudaTensorF32 CudaTensorF32::add(const CudaTensorF32& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF32 add requires identical shapes");
  }
  float* out = nullptr;
  out = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorF32 add cudaMalloc");
  const int n = static_cast<int>(numel_);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  underhfs_add_f32<<<grid, block, 0, cuda_runtime().stream()>>>(device_, other.device_, out, n);
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF32 add launch");
    cuda_runtime().synchronize("CudaTensorF32 add sync");
  } catch (...) {
    cuda_allocator().release(out, numel_ * sizeof(float));
    throw;
  }
  return CudaTensorF32(out, shape_);
}

CudaTensorF32 CudaTensorF32::mul(const CudaTensorF32& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF32 mul requires identical shapes");
  }
  float* out = nullptr;
  out = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorF32 mul cudaMalloc");
  const int n = static_cast<int>(numel_);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  underhfs_mul_f32<<<grid, block, 0, cuda_runtime().stream()>>>(device_, other.device_, out, n);
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF32 mul launch");
    cuda_runtime().synchronize("CudaTensorF32 mul sync");
  } catch (...) {
    cuda_allocator().release(out, numel_ * sizeof(float));
    throw;
  }
  return CudaTensorF32(out, shape_);
}

CudaTensorF32 CudaTensorF32::matmul(const CudaTensorF32& other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("CudaTensorF32 matmul currently requires 2D tensors");
  }
  const auto m = static_cast<int>(shape_[0]);
  const auto k = static_cast<int>(shape_[1]);
  const auto k2 = static_cast<int>(other.shape_[0]);
  const auto n = static_cast<int>(other.shape_[1]);
  if (k != k2) {
    throw std::invalid_argument("CudaTensorF32 matmul shape mismatch");
  }
  std::vector<std::size_t> out_shape = {shape_[0], other.shape_[1]};
  const auto out_numel = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  float* out = nullptr;
  out = cuda_allocator().allocate(out_numel * sizeof(float), "CudaTensorF32 matmul cudaMalloc");
  try {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasSetStream(cublas_handle(), cuda_runtime().stream()),
                 "CudaTensorF32 matmul cublasSetStream");
    check_cublas(
        cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other.device_, n,
                    device_, k, &beta, out, n),
        "CudaTensorF32 matmul cublasSgemm");
    cuda_runtime().record_launch();
    cuda_runtime().synchronize("CudaTensorF32 matmul sync");
  } catch (...) {
    cuda_allocator().release(out, out_numel * sizeof(float));
    throw;
  }
  return CudaTensorF32(out, out_shape);
}

CudaTensorF32 CudaTensorF32::sum() const {
  if (numel_ == 0) {
    throw std::invalid_argument("CudaTensorF32 sum requires at least one value");
  }
  const int block = 256;
  const float* current = device_;
  int current_n = static_cast<int>(numel_);
  std::vector<std::pair<float*, std::size_t>> temporaries;
  try {
    while (current_n > 1) {
      const int grid = (current_n + block * 2 - 1) / (block * 2);
      float* partials = nullptr;
      const auto partial_bytes = static_cast<std::size_t>(grid) * sizeof(float);
      partials = cuda_allocator().allocate(partial_bytes, "CudaTensorF32 sum partial cudaMalloc");
      temporaries.emplace_back(partials, partial_bytes);
      underhfs_sum_blocks_f32<<<grid, block, block * sizeof(float), cuda_runtime().stream()>>>(
          current, partials, current_n);
      cuda_runtime().record_launch();
      check_cuda(cudaGetLastError(), "CudaTensorF32 sum launch");
      current = partials;
      current_n = grid;
    }
    float* out = nullptr;
    out = cuda_allocator().allocate(sizeof(float), "CudaTensorF32 sum cudaMalloc");
    check_cuda(cudaMemcpyAsync(out, current, sizeof(float), cudaMemcpyDeviceToDevice,
                               cuda_runtime().stream()),
               "CudaTensorF32 sum device copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("CudaTensorF32 sum sync");
    for (auto [temporary, bytes] : temporaries) {
      cuda_allocator().release(temporary, bytes);
    }
    return CudaTensorF32(out, {});
  } catch (...) {
    for (auto [temporary, bytes] : temporaries) {
      cuda_allocator().release(temporary, bytes);
    }
    throw;
  }
}

CudaTensorF16::CudaTensorF16(const std::vector<float>& host, std::vector<std::size_t> shape)
    : shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
  if (numel_ != host.size()) {
    throw std::invalid_argument("CudaTensorF16 host size does not match shape");
  }
  float* staging = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorF16 staging cudaMalloc");
  device_ = cuda_allocator().allocate(numel_ * sizeof(__half), "CudaTensorF16 cudaMalloc");
  try {
    check_cuda(cudaMemcpyAsync(staging, host.data(), numel_ * sizeof(float), cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "CudaTensorF16 host-to-device copy");
    cuda_runtime().record_copy();
    const int block = 256;
    const int grid = (static_cast<int>(numel_) + block - 1) / block;
    underhfs_f32_to_f16<<<grid, block, 0, cuda_runtime().stream()>>>(
        staging, static_cast<__half*>(device_), static_cast<int>(numel_));
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF16 convert launch");
    cuda_runtime().synchronize("CudaTensorF16 host-to-device sync");
    cuda_allocator().release(staging, numel_ * sizeof(float));
  } catch (...) {
    cuda_allocator().release(staging, numel_ * sizeof(float));
    cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__half));
    device_ = nullptr;
    throw;
  }
}

CudaTensorF16::CudaTensorF16(void* device, std::vector<std::size_t> shape)
    : device_(device), shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
}

CudaTensorF16::~CudaTensorF16() {
  if (device_ != nullptr) {
    cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__half));
  }
}

CudaTensorF16::CudaTensorF16(CudaTensorF16&& other) noexcept
    : device_(other.device_), shape_(std::move(other.shape_)), numel_(other.numel_) {
  other.device_ = nullptr;
  other.numel_ = 0;
}

CudaTensorF16& CudaTensorF16::operator=(CudaTensorF16&& other) noexcept {
  if (this != &other) {
    if (device_ != nullptr) {
      cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__half));
    }
    device_ = other.device_;
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    other.device_ = nullptr;
    other.numel_ = 0;
  }
  return *this;
}

const std::vector<std::size_t>& CudaTensorF16::shape() const { return shape_; }

std::size_t CudaTensorF16::numel() const { return numel_; }

std::vector<float> CudaTensorF16::to_host() const {
  float* staging = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorF16 to_host staging");
  std::vector<float> host(numel_, 0.0f);
  try {
    const int block = 256;
    const int grid = (static_cast<int>(numel_) + block - 1) / block;
    underhfs_f16_to_f32<<<grid, block, 0, cuda_runtime().stream()>>>(
        static_cast<const __half*>(device_), staging, static_cast<int>(numel_));
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF16 to_host convert launch");
    check_cuda(cudaMemcpyAsync(host.data(), staging, numel_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "CudaTensorF16 device-to-host copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("CudaTensorF16 device-to-host sync");
    cuda_allocator().release(staging, numel_ * sizeof(float));
    return host;
  } catch (...) {
    cuda_allocator().release(staging, numel_ * sizeof(float));
    throw;
  }
}

CudaTensorF16 CudaTensorF16::add(const CudaTensorF16& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF16 add requires identical shapes");
  }
  void* out = cuda_allocator().allocate(numel_ * sizeof(__half), "CudaTensorF16 add cudaMalloc");
  const int block = 256;
  const int grid = (static_cast<int>(numel_) + block - 1) / block;
  underhfs_add_f16<<<grid, block, 0, cuda_runtime().stream()>>>(
      static_cast<const __half*>(device_), static_cast<const __half*>(other.device_),
      static_cast<__half*>(out), static_cast<int>(numel_));
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF16 add launch");
    cuda_runtime().synchronize("CudaTensorF16 add sync");
  } catch (...) {
    cuda_allocator().release(static_cast<float*>(out), numel_ * sizeof(__half));
    throw;
  }
  return CudaTensorF16(out, shape_);
}

CudaTensorF16 CudaTensorF16::mul(const CudaTensorF16& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF16 mul requires identical shapes");
  }
  void* out = cuda_allocator().allocate(numel_ * sizeof(__half), "CudaTensorF16 mul cudaMalloc");
  const int block = 256;
  const int grid = (static_cast<int>(numel_) + block - 1) / block;
  underhfs_mul_f16<<<grid, block, 0, cuda_runtime().stream()>>>(
      static_cast<const __half*>(device_), static_cast<const __half*>(other.device_),
      static_cast<__half*>(out), static_cast<int>(numel_));
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorF16 mul launch");
    cuda_runtime().synchronize("CudaTensorF16 mul sync");
  } catch (...) {
    cuda_allocator().release(static_cast<float*>(out), numel_ * sizeof(__half));
    throw;
  }
  return CudaTensorF16(out, shape_);
}

CudaTensorBF16::CudaTensorBF16(const std::vector<float>& host, std::vector<std::size_t> shape)
    : shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
  if (numel_ != host.size()) {
    throw std::invalid_argument("CudaTensorBF16 host size does not match shape");
  }
  float* staging = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorBF16 staging cudaMalloc");
  device_ = cuda_allocator().allocate(numel_ * sizeof(__nv_bfloat16), "CudaTensorBF16 cudaMalloc");
  try {
    check_cuda(cudaMemcpyAsync(staging, host.data(), numel_ * sizeof(float), cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "CudaTensorBF16 host-to-device copy");
    cuda_runtime().record_copy();
    const int block = 256;
    const int grid = (static_cast<int>(numel_) + block - 1) / block;
    underhfs_f32_to_bf16<<<grid, block, 0, cuda_runtime().stream()>>>(
        staging, static_cast<__nv_bfloat16*>(device_), static_cast<int>(numel_));
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorBF16 convert launch");
    cuda_runtime().synchronize("CudaTensorBF16 host-to-device sync");
    cuda_allocator().release(staging, numel_ * sizeof(float));
  } catch (...) {
    cuda_allocator().release(staging, numel_ * sizeof(float));
    cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__nv_bfloat16));
    device_ = nullptr;
    throw;
  }
}

CudaTensorBF16::CudaTensorBF16(void* device, std::vector<std::size_t> shape)
    : device_(device), shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
}

CudaTensorBF16::~CudaTensorBF16() {
  if (device_ != nullptr) {
    cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__nv_bfloat16));
  }
}

CudaTensorBF16::CudaTensorBF16(CudaTensorBF16&& other) noexcept
    : device_(other.device_), shape_(std::move(other.shape_)), numel_(other.numel_) {
  other.device_ = nullptr;
  other.numel_ = 0;
}

CudaTensorBF16& CudaTensorBF16::operator=(CudaTensorBF16&& other) noexcept {
  if (this != &other) {
    if (device_ != nullptr) {
      cuda_allocator().release(static_cast<float*>(device_), numel_ * sizeof(__nv_bfloat16));
    }
    device_ = other.device_;
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    other.device_ = nullptr;
    other.numel_ = 0;
  }
  return *this;
}

const std::vector<std::size_t>& CudaTensorBF16::shape() const { return shape_; }

std::size_t CudaTensorBF16::numel() const { return numel_; }

std::vector<float> CudaTensorBF16::to_host() const {
  float* staging = cuda_allocator().allocate(numel_ * sizeof(float), "CudaTensorBF16 to_host staging");
  std::vector<float> host(numel_, 0.0f);
  try {
    const int block = 256;
    const int grid = (static_cast<int>(numel_) + block - 1) / block;
    underhfs_bf16_to_f32<<<grid, block, 0, cuda_runtime().stream()>>>(
        static_cast<const __nv_bfloat16*>(device_), staging, static_cast<int>(numel_));
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorBF16 to_host convert launch");
    check_cuda(cudaMemcpyAsync(host.data(), staging, numel_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "CudaTensorBF16 device-to-host copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("CudaTensorBF16 device-to-host sync");
    cuda_allocator().release(staging, numel_ * sizeof(float));
    return host;
  } catch (...) {
    cuda_allocator().release(staging, numel_ * sizeof(float));
    throw;
  }
}

CudaTensorBF16 CudaTensorBF16::add(const CudaTensorBF16& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorBF16 add requires identical shapes");
  }
  void* out = cuda_allocator().allocate(numel_ * sizeof(__nv_bfloat16), "CudaTensorBF16 add cudaMalloc");
  const int block = 256;
  const int grid = (static_cast<int>(numel_) + block - 1) / block;
  underhfs_add_bf16<<<grid, block, 0, cuda_runtime().stream()>>>(
      static_cast<const __nv_bfloat16*>(device_), static_cast<const __nv_bfloat16*>(other.device_),
      static_cast<__nv_bfloat16*>(out), static_cast<int>(numel_));
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorBF16 add launch");
    cuda_runtime().synchronize("CudaTensorBF16 add sync");
  } catch (...) {
    cuda_allocator().release(static_cast<float*>(out), numel_ * sizeof(__nv_bfloat16));
    throw;
  }
  return CudaTensorBF16(out, shape_);
}

CudaTensorBF16 CudaTensorBF16::mul(const CudaTensorBF16& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorBF16 mul requires identical shapes");
  }
  void* out = cuda_allocator().allocate(numel_ * sizeof(__nv_bfloat16), "CudaTensorBF16 mul cudaMalloc");
  const int block = 256;
  const int grid = (static_cast<int>(numel_) + block - 1) / block;
  underhfs_mul_bf16<<<grid, block, 0, cuda_runtime().stream()>>>(
      static_cast<const __nv_bfloat16*>(device_), static_cast<const __nv_bfloat16*>(other.device_),
      static_cast<__nv_bfloat16*>(out), static_cast<int>(numel_));
  try {
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "CudaTensorBF16 mul launch");
    cuda_runtime().synchronize("CudaTensorBF16 mul sync");
  } catch (...) {
    cuda_allocator().release(static_cast<float*>(out), numel_ * sizeof(__nv_bfloat16));
    throw;
  }
  return CudaTensorBF16(out, shape_);
}

std::vector<float> cuda_add_f32_host(const std::vector<float>& left,
                                     const std::vector<float>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("cuda_add_f32_host requires equal vector sizes");
  }
  const auto n = static_cast<int>(left.size());
  const auto bytes = left.size() * sizeof(float);
  std::vector<float> out(left.size(), 0.0f);
  float* d_left = nullptr;
  float* d_right = nullptr;
  float* d_out = nullptr;

  check_cuda(cudaMalloc(&d_left, bytes), "cudaMalloc(left)");
  check_cuda(cudaMalloc(&d_right, bytes), "cudaMalloc(right)");
  check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc(out)");

  try {
    check_cuda(cudaMemcpyAsync(d_left, left.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cudaMemcpy(left)");
    cuda_runtime().record_copy();
    check_cuda(cudaMemcpyAsync(d_right, right.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cudaMemcpy(right)");
    cuda_runtime().record_copy();
    const int block = 256;
    const int grid = (n + block - 1) / block;
    underhfs_add_f32<<<grid, block, 0, cuda_runtime().stream()>>>(d_left, d_right, d_out, n);
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "underhfs_add_f32 launch");
    check_cuda(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cudaMemcpy(out)");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("underhfs_add_f32 sync");
  } catch (...) {
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_out);
    throw;
  }

  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(d_out);
  return out;
}

std::unordered_map<std::string, std::vector<float>> cuda_fused_adamw_f32_host(
    const std::vector<float>& param,
    const std::vector<float>& grad,
    const std::vector<float>& m,
    const std::vector<float>& v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step) {
  if (param.size() != grad.size() || param.size() != m.size() || param.size() != v.size()) {
    throw std::invalid_argument("cuda_fused_adamw_f32_host requires equal vector sizes");
  }
  if (step <= 0) {
    throw std::invalid_argument("cuda_fused_adamw_f32_host step must be positive");
  }
  const auto n = static_cast<int>(param.size());
  const auto bytes = param.size() * sizeof(float);
  std::vector<float> out_param(param.size(), 0.0f);
  std::vector<float> out_m(param.size(), 0.0f);
  std::vector<float> out_v(param.size(), 0.0f);
  float* d_param = nullptr;
  float* d_grad = nullptr;
  float* d_m = nullptr;
  float* d_v = nullptr;

  d_param = cuda_allocator().allocate(bytes, "cuda_fused_adamw param cudaMalloc");
  d_grad = cuda_allocator().allocate(bytes, "cuda_fused_adamw grad cudaMalloc");
  d_m = cuda_allocator().allocate(bytes, "cuda_fused_adamw m cudaMalloc");
  d_v = cuda_allocator().allocate(bytes, "cuda_fused_adamw v cudaMalloc");

  try {
    check_cuda(cudaMemcpyAsync(d_param, param.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_fused_adamw param copy");
    check_cuda(cudaMemcpyAsync(d_grad, grad.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_fused_adamw grad copy");
    check_cuda(cudaMemcpyAsync(d_m, m.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_fused_adamw m copy");
    check_cuda(cudaMemcpyAsync(d_v, v.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_fused_adamw v copy");
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    const int block = 256;
    const int grid = (n + block - 1) / block;
    underhfs_fused_adamw_f32<<<grid, block, 0, cuda_runtime().stream()>>>(
        d_param, d_grad, d_m, d_v, n, lr, beta1, beta2, eps, weight_decay,
        1.0f - std::pow(beta1, static_cast<float>(step)),
        1.0f - std::pow(beta2, static_cast<float>(step)));
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "cuda_fused_adamw launch");
    check_cuda(cudaMemcpyAsync(out_param.data(), d_param, bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cuda_fused_adamw param back copy");
    check_cuda(cudaMemcpyAsync(out_m.data(), d_m, bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cuda_fused_adamw m back copy");
    check_cuda(cudaMemcpyAsync(out_v.data(), d_v, bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cuda_fused_adamw v back copy");
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("cuda_fused_adamw sync");
  } catch (...) {
    cuda_allocator().release(d_param, bytes);
    cuda_allocator().release(d_grad, bytes);
    cuda_allocator().release(d_m, bytes);
    cuda_allocator().release(d_v, bytes);
    throw;
  }

  cuda_allocator().release(d_param, bytes);
  cuda_allocator().release(d_grad, bytes);
  cuda_allocator().release(d_m, bytes);
  cuda_allocator().release(d_v, bytes);
  return {{"param", out_param}, {"m", out_m}, {"v", out_v}};
}

std::vector<float> cuda_attention_f32_host(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int tokens,
    int features,
    float scale,
    bool causal) {
  if (tokens <= 0 || features <= 0) {
    throw std::invalid_argument("cuda_attention_f32_host requires positive tokens and features");
  }
  const auto expected = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(features);
  if (q.size() != expected || k.size() != expected || v.size() != expected) {
    throw std::invalid_argument("cuda_attention_f32_host tensor sizes must equal tokens * features");
  }
  const auto bytes = expected * sizeof(float);
  std::vector<float> out(expected, 0.0f);
  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_out = nullptr;

  d_q = cuda_allocator().allocate(bytes, "cuda_attention q cudaMalloc");
  d_k = cuda_allocator().allocate(bytes, "cuda_attention k cudaMalloc");
  d_v = cuda_allocator().allocate(bytes, "cuda_attention v cudaMalloc");
  d_out = cuda_allocator().allocate(bytes, "cuda_attention out cudaMalloc");

  try {
    check_cuda(cudaMemcpyAsync(d_q, q.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_attention q copy");
    check_cuda(cudaMemcpyAsync(d_k, k.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_attention k copy");
    check_cuda(cudaMemcpyAsync(d_v, v.data(), bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cuda_attention v copy");
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    underhfs_attention_f32<<<tokens, features, 0, cuda_runtime().stream()>>>(
        d_q, d_k, d_v, d_out, tokens, features, scale, causal ? 1 : 0);
    cuda_runtime().record_launch();
    check_cuda(cudaGetLastError(), "cuda_attention launch");
    check_cuda(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cuda_attention out copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("cuda_attention sync");
  } catch (...) {
    cuda_allocator().release(d_q, bytes);
    cuda_allocator().release(d_k, bytes);
    cuda_allocator().release(d_v, bytes);
    cuda_allocator().release(d_out, bytes);
    throw;
  }
  cuda_allocator().release(d_q, bytes);
  cuda_allocator().release(d_k, bytes);
  cuda_allocator().release(d_v, bytes);
  cuda_allocator().release(d_out, bytes);
  return out;
}

std::vector<float> cudnn_conv2d_forward_f32_host(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int batch,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {
  if (batch <= 0 || in_channels <= 0 || height <= 0 || width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0) {
    throw std::invalid_argument("cudnn_conv2d_forward_f32 received invalid dimensions");
  }
  const int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  if (out_h <= 0 || out_w <= 0) {
    throw std::invalid_argument("cudnn_conv2d_forward_f32 produced an empty output shape");
  }
  const std::size_t expected_input = static_cast<std::size_t>(batch) * in_channels * height * width;
  const std::size_t expected_weight = static_cast<std::size_t>(out_channels) * in_channels * kernel_h * kernel_w;
  if (input.size() != expected_input || weight.size() != expected_weight) {
    throw std::invalid_argument("cudnn_conv2d_forward_f32 payload size does not match tensor shapes");
  }
  if (!bias.empty() && bias.size() != static_cast<std::size_t>(out_channels)) {
    throw std::invalid_argument("cudnn_conv2d_forward_f32 bias size must match out_channels");
  }
#ifdef UNDERHFS_WITH_CUDNN
  const std::size_t input_bytes = input.size() * sizeof(float);
  const std::size_t weight_bytes = weight.size() * sizeof(float);
  const std::size_t bias_bytes = bias.size() * sizeof(float);
  const std::size_t output_size = static_cast<std::size_t>(batch) * out_channels * out_h * out_w;
  const std::size_t output_bytes = output_size * sizeof(float);
  std::vector<float> out(output_size, 0.0f);
  float* d_input = nullptr;
  float* d_weight = nullptr;
  float* d_bias = nullptr;
  float* d_out = nullptr;
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t input_desc = nullptr;
  cudnnFilterDescriptor_t weight_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnTensorDescriptor_t output_desc = nullptr;
  cudnnTensorDescriptor_t bias_desc = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;

  d_input = cuda_allocator().allocate(input_bytes, "cudnn conv2d input cudaMalloc");
  d_weight = cuda_allocator().allocate(weight_bytes, "cudnn conv2d weight cudaMalloc");
  d_out = cuda_allocator().allocate(output_bytes, "cudnn conv2d out cudaMalloc");
  if (!bias.empty()) {
    d_bias = cuda_allocator().allocate(bias_bytes, "cudnn conv2d bias cudaMalloc");
  }
  try {
    check_cuda(cudaMemcpyAsync(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cudnn conv2d input copy");
    check_cuda(cudaMemcpyAsync(d_weight, weight.data(), weight_bytes, cudaMemcpyHostToDevice,
                               cuda_runtime().stream()),
               "cudnn conv2d weight copy");
    cuda_runtime().record_copy();
    cuda_runtime().record_copy();
    if (!bias.empty()) {
      check_cuda(cudaMemcpyAsync(d_bias, bias.data(), bias_bytes, cudaMemcpyHostToDevice,
                                 cuda_runtime().stream()),
                 "cudnn conv2d bias copy");
      cuda_runtime().record_copy();
    }
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(handle, cuda_runtime().stream()), "cudnnSetStream");
    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor(input)");
    check_cudnn(cudnnCreateFilterDescriptor(&weight_desc), "cudnnCreateFilterDescriptor(weight)");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor(output)");
    check_cudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, in_channels, height, width),
                "cudnnSetTensor4dDescriptor(input)");
    check_cudnn(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_h, kernel_w),
                "cudnnSetFilter4dDescriptor(weight)");
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor");
    check_cudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, out_channels, out_h, out_w),
                "cudnnSetTensor4dDescriptor(output)");
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, weight_desc, conv_desc,
                                                        output_desc, algo, &workspace_bytes),
                "cudnnGetConvolutionForwardWorkspaceSize");
    if (workspace_bytes > 0) {
      check_cuda(cudaMalloc(&workspace, workspace_bytes), "cudnn conv2d workspace cudaMalloc");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cudnn(cudnnConvolutionForward(handle, &alpha, input_desc, d_input, weight_desc, d_weight,
                                        conv_desc, algo, workspace, workspace_bytes, &beta,
                                        output_desc, d_out),
                "cudnnConvolutionForward");
    cuda_runtime().record_launch();
    if (!bias.empty()) {
      check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor(bias)");
      check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                             1, out_channels, 1, 1),
                  "cudnnSetTensor4dDescriptor(bias)");
      check_cudnn(cudnnAddTensor(handle, &alpha, bias_desc, d_bias, &alpha, output_desc, d_out),
                  "cudnnAddTensor(bias)");
      cuda_runtime().record_launch();
    }
    check_cuda(cudaMemcpyAsync(out.data(), d_out, output_bytes, cudaMemcpyDeviceToHost,
                               cuda_runtime().stream()),
               "cudnn conv2d output copy");
    cuda_runtime().record_copy();
    cuda_runtime().synchronize("cudnn conv2d sync");
  } catch (...) {
    if (workspace != nullptr) cudaFree(workspace);
    if (bias_desc != nullptr) cudnnDestroyTensorDescriptor(bias_desc);
    if (output_desc != nullptr) cudnnDestroyTensorDescriptor(output_desc);
    if (conv_desc != nullptr) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (weight_desc != nullptr) cudnnDestroyFilterDescriptor(weight_desc);
    if (input_desc != nullptr) cudnnDestroyTensorDescriptor(input_desc);
    if (handle != nullptr) cudnnDestroy(handle);
    cuda_allocator().release(d_input, input_bytes);
    cuda_allocator().release(d_weight, weight_bytes);
    if (d_bias != nullptr) cuda_allocator().release(d_bias, bias_bytes);
    cuda_allocator().release(d_out, output_bytes);
    throw;
  }
  if (workspace != nullptr) cudaFree(workspace);
  if (bias_desc != nullptr) cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(weight_desc);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroy(handle);
  cuda_allocator().release(d_input, input_bytes);
  cuda_allocator().release(d_weight, weight_bytes);
  if (d_bias != nullptr) cuda_allocator().release(d_bias, bias_bytes);
  cuda_allocator().release(d_out, output_bytes);
  return out;
#else
  std::vector<float> out(static_cast<std::size_t>(batch) * out_channels * out_h * out_w, 0.0f);
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float acc = bias.empty() ? 0.0f : bias[oc];
          for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                const int ih = oh * stride_h + kh - pad_h;
                const int iw = ow * stride_w + kw - pad_w;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const std::size_t input_index =
                      ((static_cast<std::size_t>(n) * in_channels + ic) * height + ih) * width + iw;
                  const std::size_t weight_index =
                      ((static_cast<std::size_t>(oc) * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                  acc += input[input_index] * weight[weight_index];
                }
              }
            }
          }
          out[((static_cast<std::size_t>(n) * out_channels + oc) * out_h + oh) * out_w + ow] = acc;
        }
      }
    }
  }
  return out;
#endif
}

std::unordered_map<std::string, std::size_t> cuda_allocator_stats() {
  return cuda_allocator().stats();
}

void cuda_empty_cache() { cuda_allocator().empty_cache(); }

std::unordered_map<std::string, std::size_t> cuda_stream_stats() {
  return cuda_runtime().stats();
}

void cuda_synchronize() { cuda_runtime().synchronize("cuda_synchronize"); }

}  // namespace underhfs
