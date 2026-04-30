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
  std::size_t numel() const;
  std::string repr() const;

 private:
  std::vector<double> storage_;
  std::vector<std::size_t> shape_;
};

}  // namespace underhfs
