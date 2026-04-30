#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <numeric>
#include <mutex>
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

std::unordered_map<std::string, std::size_t> cuda_allocator_stats() {
  return cuda_allocator().stats();
}

void cuda_empty_cache() { cuda_allocator().empty_cache(); }

std::unordered_map<std::string, std::size_t> cuda_stream_stats() {
  return cuda_runtime().stats();
}

void cuda_synchronize() { cuda_runtime().synchronize("cuda_synchronize"); }

}  // namespace underhfs
