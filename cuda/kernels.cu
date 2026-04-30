#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <numeric>
#include <stdexcept>
#include <string>
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

}  // namespace

CudaTensorF32::CudaTensorF32(const std::vector<float>& host, std::vector<std::size_t> shape)
    : shape_(std::move(shape)) {
  numel_ = std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
  if (numel_ != host.size()) {
    throw std::invalid_argument("CudaTensorF32 host size does not match shape");
  }
  check_cuda(cudaMalloc(&device_, numel_ * sizeof(float)), "CudaTensorF32 cudaMalloc");
  try {
    check_cuda(cudaMemcpy(device_, host.data(), numel_ * sizeof(float), cudaMemcpyHostToDevice),
               "CudaTensorF32 host-to-device copy");
  } catch (...) {
    cudaFree(device_);
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
    cudaFree(device_);
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
      cudaFree(device_);
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
  check_cuda(cudaMemcpy(host.data(), device_, numel_ * sizeof(float), cudaMemcpyDeviceToHost),
             "CudaTensorF32 device-to-host copy");
  return host;
}

CudaTensorF32 CudaTensorF32::add(const CudaTensorF32& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF32 add requires identical shapes");
  }
  float* out = nullptr;
  check_cuda(cudaMalloc(&out, numel_ * sizeof(float)), "CudaTensorF32 add cudaMalloc");
  const int n = static_cast<int>(numel_);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  underhfs_add_f32<<<grid, block>>>(device_, other.device_, out, n);
  try {
    check_cuda(cudaGetLastError(), "CudaTensorF32 add launch");
    check_cuda(cudaDeviceSynchronize(), "CudaTensorF32 add sync");
  } catch (...) {
    cudaFree(out);
    throw;
  }
  return CudaTensorF32(out, shape_);
}

CudaTensorF32 CudaTensorF32::mul(const CudaTensorF32& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("CudaTensorF32 mul requires identical shapes");
  }
  float* out = nullptr;
  check_cuda(cudaMalloc(&out, numel_ * sizeof(float)), "CudaTensorF32 mul cudaMalloc");
  const int n = static_cast<int>(numel_);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  underhfs_mul_f32<<<grid, block>>>(device_, other.device_, out, n);
  try {
    check_cuda(cudaGetLastError(), "CudaTensorF32 mul launch");
    check_cuda(cudaDeviceSynchronize(), "CudaTensorF32 mul sync");
  } catch (...) {
    cudaFree(out);
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
  check_cuda(cudaMalloc(&out, out_numel * sizeof(float)), "CudaTensorF32 matmul cudaMalloc");
  try {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other.device_, n,
                    device_, k, &beta, out, n),
        "CudaTensorF32 matmul cublasSgemm");
    check_cuda(cudaDeviceSynchronize(), "CudaTensorF32 matmul sync");
  } catch (...) {
    cudaFree(out);
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
  std::vector<float*> temporaries;
  try {
    while (current_n > 1) {
      const int grid = (current_n + block * 2 - 1) / (block * 2);
      float* partials = nullptr;
      check_cuda(cudaMalloc(&partials, static_cast<std::size_t>(grid) * sizeof(float)),
                 "CudaTensorF32 sum partial cudaMalloc");
      temporaries.push_back(partials);
      underhfs_sum_blocks_f32<<<grid, block, block * sizeof(float)>>>(current, partials, current_n);
      check_cuda(cudaGetLastError(), "CudaTensorF32 sum launch");
      current = partials;
      current_n = grid;
    }
    float* out = nullptr;
    check_cuda(cudaMalloc(&out, sizeof(float)), "CudaTensorF32 sum cudaMalloc");
    check_cuda(cudaMemcpy(out, current, sizeof(float), cudaMemcpyDeviceToDevice),
               "CudaTensorF32 sum device copy");
    check_cuda(cudaDeviceSynchronize(), "CudaTensorF32 sum sync");
    for (float* temporary : temporaries) {
      cudaFree(temporary);
    }
    return CudaTensorF32(out, {});
  } catch (...) {
    for (float* temporary : temporaries) {
      cudaFree(temporary);
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
    check_cuda(cudaMemcpy(d_left, left.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(left)");
    check_cuda(cudaMemcpy(d_right, right.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(right)");
    const int block = 256;
    const int grid = (n + block - 1) / block;
    underhfs_add_f32<<<grid, block>>>(d_left, d_right, d_out, n);
    check_cuda(cudaGetLastError(), "underhfs_add_f32 launch");
    check_cuda(cudaDeviceSynchronize(), "underhfs_add_f32 sync");
    check_cuda(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(out)");
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

}  // namespace underhfs
