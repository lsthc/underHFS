#pragma once

#include <unordered_map>
#include <string>
#include <vector>

namespace underhfs {

class CudaTensorF32 {
 public:
  CudaTensorF32(const std::vector<float>& host, std::vector<std::size_t> shape);
  ~CudaTensorF32();

  CudaTensorF32(const CudaTensorF32&) = delete;
  CudaTensorF32& operator=(const CudaTensorF32&) = delete;
  CudaTensorF32(CudaTensorF32&& other) noexcept;
  CudaTensorF32& operator=(CudaTensorF32&& other) noexcept;

  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::vector<float> to_host() const;
  CudaTensorF32 add(const CudaTensorF32& other) const;
  CudaTensorF32 mul(const CudaTensorF32& other) const;
  CudaTensorF32 matmul(const CudaTensorF32& other) const;
  CudaTensorF32 sum() const;

 private:
  CudaTensorF32(float* device, std::vector<std::size_t> shape);

  float* device_ = nullptr;
  std::vector<std::size_t> shape_;
  std::size_t numel_ = 0;
};

class CudaTensorF16 {
 public:
  CudaTensorF16(const std::vector<float>& host, std::vector<std::size_t> shape);
  ~CudaTensorF16();

  CudaTensorF16(const CudaTensorF16&) = delete;
  CudaTensorF16& operator=(const CudaTensorF16&) = delete;
  CudaTensorF16(CudaTensorF16&& other) noexcept;
  CudaTensorF16& operator=(CudaTensorF16&& other) noexcept;

  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::vector<float> to_host() const;
  CudaTensorF16 add(const CudaTensorF16& other) const;
  CudaTensorF16 mul(const CudaTensorF16& other) const;

 private:
  CudaTensorF16(void* device, std::vector<std::size_t> shape);

  void* device_ = nullptr;
  std::vector<std::size_t> shape_;
  std::size_t numel_ = 0;
};

class CudaTensorBF16 {
 public:
  CudaTensorBF16(const std::vector<float>& host, std::vector<std::size_t> shape);
  ~CudaTensorBF16();

  CudaTensorBF16(const CudaTensorBF16&) = delete;
  CudaTensorBF16& operator=(const CudaTensorBF16&) = delete;
  CudaTensorBF16(CudaTensorBF16&& other) noexcept;
  CudaTensorBF16& operator=(CudaTensorBF16&& other) noexcept;

  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::vector<float> to_host() const;
  CudaTensorBF16 add(const CudaTensorBF16& other) const;
  CudaTensorBF16 mul(const CudaTensorBF16& other) const;

 private:
  CudaTensorBF16(void* device, std::vector<std::size_t> shape);

  void* device_ = nullptr;
  std::vector<std::size_t> shape_;
  std::size_t numel_ = 0;
};

std::vector<float> cuda_add_f32_host(const std::vector<float>& left,
                                     const std::vector<float>& right);
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
    int step);
std::vector<float> cuda_attention_f32_host(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int tokens,
    int features,
    float scale,
    bool causal);
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
    int pad_w);
std::vector<float> cudnn_conv2d_backward_input_f32_host(
    const std::vector<float>& grad_output,
    const std::vector<float>& weight,
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
    int pad_w);
std::vector<float> cudnn_conv2d_backward_weight_f32_host(
    const std::vector<float>& input,
    const std::vector<float>& grad_output,
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
    int pad_w);
std::unordered_map<std::string, std::size_t> cuda_allocator_stats();
void cuda_empty_cache();
std::unordered_map<std::string, std::size_t> cuda_stream_stats();
void cuda_synchronize();

}  // namespace underhfs
