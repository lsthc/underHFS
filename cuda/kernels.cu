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

namespace underhfs {

namespace {

void check_cuda(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
  }
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
