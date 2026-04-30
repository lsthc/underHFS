#include "tensor_core.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace underhfs {

TensorCore::TensorCore(std::vector<double> storage, std::vector<std::size_t> shape)
    : storage_(std::move(storage)), shape_(std::move(shape)) {
  std::size_t expected = 1;
  for (auto dim : shape_) {
    expected *= dim;
  }
  if (expected != storage_.size()) {
    throw std::invalid_argument("storage size does not match shape");
  }
}

const std::vector<double>& TensorCore::storage() const { return storage_; }

const std::vector<std::size_t>& TensorCore::shape() const { return shape_; }

std::size_t TensorCore::numel() const { return storage_.size(); }

std::string TensorCore::repr() const {
  std::ostringstream out;
  out << "TensorCore(numel=" << storage_.size() << ")";
  return out.str();
}

}  // namespace underhfs
