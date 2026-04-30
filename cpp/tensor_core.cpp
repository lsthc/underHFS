#include "tensor_core.hpp"

#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace underhfs {

TensorCore::TensorCore(std::vector<double> storage, std::vector<std::size_t> shape)
    : storage_(std::move(storage)), shape_(std::move(shape)) {
  const auto expected = shape_numel(shape_);
  if (expected != storage_.size()) {
    throw std::invalid_argument("storage size does not match shape");
  }
}

const std::vector<double>& TensorCore::storage() const { return storage_; }

const std::vector<std::size_t>& TensorCore::shape() const { return shape_; }

std::vector<std::size_t> TensorCore::strides() const { return contiguous_strides(shape_); }

std::size_t TensorCore::numel() const { return storage_.size(); }

TensorCore TensorCore::add(const TensorCore& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("native add requires identical shapes");
  }
  std::vector<double> out(storage_.size());
  for (std::size_t i = 0; i < storage_.size(); ++i) {
    out[i] = storage_[i] + other.storage_[i];
  }
  return TensorCore(std::move(out), shape_);
}

TensorCore TensorCore::mul(const TensorCore& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("native mul requires identical shapes");
  }
  std::vector<double> out(storage_.size());
  for (std::size_t i = 0; i < storage_.size(); ++i) {
    out[i] = storage_[i] * other.storage_[i];
  }
  return TensorCore(std::move(out), shape_);
}

TensorCore TensorCore::matmul(const TensorCore& other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("native matmul requires two 2D tensors");
  }
  const auto m = shape_[0];
  const auto k = shape_[1];
  const auto k2 = other.shape_[0];
  const auto n = other.shape_[1];
  if (k != k2) {
    throw std::invalid_argument("native matmul shape mismatch");
  }
  std::vector<double> out(m * n, 0.0);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double acc = 0.0;
      for (std::size_t inner = 0; inner < k; ++inner) {
        acc += storage_[row * k + inner] * other.storage_[inner * n + col];
      }
      out[row * n + col] = acc;
    }
  }
  return TensorCore(std::move(out), {m, n});
}

TensorCore TensorCore::sum() const {
  const auto value = std::accumulate(storage_.begin(), storage_.end(), 0.0);
  return TensorCore({value}, {});
}

std::string TensorCore::repr() const {
  std::ostringstream out;
  out << "TensorCore(shape=[";
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape_[i];
  }
  out << "], numel=" << storage_.size() << ")";
  return out.str();
}

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape) {
  std::vector<std::size_t> strides(shape.size(), 1);
  std::size_t stride = 1;
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    const auto index = static_cast<std::size_t>(std::distance(it, shape.rend()) - 1);
    strides[index] = stride;
    stride *= *it;
  }
  return strides;
}

std::size_t shape_numel(const std::vector<std::size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1),
                         std::multiplies<>());
}

}  // namespace underhfs
