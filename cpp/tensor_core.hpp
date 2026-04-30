#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace underhfs {

enum class DType { fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, int8, int4 };
enum class Layout { dense, sparse, quantized };

struct Device {
  std::string kind = "cpu";
  int index = -1;
};

class TensorCore {
 public:
  TensorCore(std::vector<double> storage, std::vector<std::size_t> shape);

  const std::vector<double>& storage() const;
  const std::vector<std::size_t>& shape() const;
  std::vector<std::size_t> strides() const;
  std::size_t numel() const;
  TensorCore add(const TensorCore& other) const;
  TensorCore mul(const TensorCore& other) const;
  TensorCore matmul(const TensorCore& other) const;
  TensorCore sum() const;
  std::string repr() const;

 private:
  std::vector<double> storage_;
  std::vector<std::size_t> shape_;
};

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape);
std::size_t shape_numel(const std::vector<std::size_t>& shape);

}  // namespace underhfs
