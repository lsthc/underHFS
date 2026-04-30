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

std::vector<float> cuda_add_f32_host(const std::vector<float>& left,
                                     const std::vector<float>& right);
std::unordered_map<std::string, std::size_t> cuda_allocator_stats();
void cuda_empty_cache();
std::unordered_map<std::string, std::size_t> cuda_stream_stats();
void cuda_synchronize();

}  // namespace underhfs
