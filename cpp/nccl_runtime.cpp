#include "nccl_runtime.hpp"

#ifdef UNDERHFS_WITH_NCCL
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#endif

namespace underhfs {

#ifdef UNDERHFS_WITH_NCCL
namespace {

void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

void check_nccl(ncclResult_t status, const char* context) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(context) + ": " + ncclGetErrorString(status));
  }
}

std::string to_hex(const char* bytes, std::size_t size) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < size; ++i) {
    out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(bytes[i]));
  }
  return out.str();
}

ncclUniqueId from_hex(const std::string& hex) {
  if (hex.size() != NCCL_UNIQUE_ID_BYTES * 2) {
    throw std::invalid_argument("NCCL unique id hex length must equal NCCL_UNIQUE_ID_BYTES * 2");
  }
  ncclUniqueId id{};
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    const std::string byte = hex.substr(static_cast<std::size_t>(i) * 2, 2);
    id.internal[i] = static_cast<char>(std::stoul(byte, nullptr, 16));
  }
  return id;
}

std::vector<float> copy_collective(
    ncclComm_t comm,
    const std::vector<float>& input,
    std::size_t output_count,
    const char* context,
    const std::function<void(const float*, float*, cudaStream_t)>& collective) {
  float* d_in = nullptr;
  float* d_out = nullptr;
  const std::size_t in_bytes = input.size() * sizeof(float);
  const std::size_t out_bytes = output_count * sizeof(float);
  std::vector<float> output(output_count, 0.0f);
  check_cuda(cudaMalloc(&d_in, in_bytes), context);
  check_cuda(cudaMalloc(&d_out, out_bytes), context);
  try {
    check_cuda(cudaMemcpy(d_in, input.data(), in_bytes, cudaMemcpyHostToDevice), context);
    cudaStream_t stream = nullptr;
    collective(d_in, d_out, stream);
    check_cuda(cudaStreamSynchronize(stream), context);
    check_cuda(cudaMemcpy(output.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), context);
  } catch (...) {
    cudaFree(d_in);
    cudaFree(d_out);
    throw;
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return output;
}

}  // namespace

NcclProcessGroup::NcclProcessGroup(int rank, int world_size, const std::string& unique_id_hex)
    : rank_(rank), world_size_(world_size) {
  if (world_size_ <= 0) {
    throw std::invalid_argument("NCCL world_size must be positive");
  }
  if (rank_ < 0 || rank_ >= world_size_) {
    throw std::invalid_argument("NCCL rank must be in [0, world_size)");
  }
  ncclUniqueId id{};
  if (unique_id_hex.empty()) {
    if (world_size_ != 1) {
      throw std::invalid_argument("multi-rank NCCL process groups require a shared unique id hex");
    }
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  } else {
    id = from_hex(unique_id_hex);
  }
  ncclComm_t comm = nullptr;
  check_nccl(ncclCommInitRank(&comm, world_size_, id, rank_), "ncclCommInitRank");
  comm_ = comm;
}

NcclProcessGroup::~NcclProcessGroup() {
  if (comm_ != nullptr) {
    ncclCommDestroy(static_cast<ncclComm_t>(comm_));
    comm_ = nullptr;
  }
}

int NcclProcessGroup::rank() const { return rank_; }

int NcclProcessGroup::world_size() const { return world_size_; }

void NcclProcessGroup::barrier() const {
  static_cast<void>(all_reduce_sum({1.0f}));
}

std::vector<float> NcclProcessGroup::all_reduce_sum(const std::vector<float>& value) const {
  if (value.empty()) return {};
  auto comm = static_cast<ncclComm_t>(comm_);
  return copy_collective(comm, value, value.size(), "ncclAllReduce", [comm, count = value.size()](
                                                                    const float* in,
                                                                    float* out,
                                                                    cudaStream_t stream) {
    check_nccl(ncclAllReduce(in, out, count, ncclFloat32, ncclSum, comm, stream), "ncclAllReduce");
  });
}

std::vector<float> NcclProcessGroup::broadcast(const std::vector<float>& value, int src) const {
  if (src < 0 || src >= world_size_) {
    throw std::invalid_argument("broadcast src must be in [0, world_size)");
  }
  if (value.empty()) return {};
  auto comm = static_cast<ncclComm_t>(comm_);
  return copy_collective(comm, value, value.size(), "ncclBroadcast", [comm, count = value.size(), src](
                                                                    const float* in,
                                                                    float* out,
                                                                    cudaStream_t stream) {
    check_nccl(ncclBroadcast(in, out, count, ncclFloat32, src, comm, stream), "ncclBroadcast");
  });
}

std::vector<float> NcclProcessGroup::reduce_scatter(const std::vector<float>& values) const {
  if (values.empty()) return {};
  if (values.size() % static_cast<std::size_t>(world_size_) != 0) {
    throw std::invalid_argument("reduce_scatter input length must be divisible by world_size");
  }
  const std::size_t out_count = values.size() / static_cast<std::size_t>(world_size_);
  auto comm = static_cast<ncclComm_t>(comm_);
  return copy_collective(comm, values, out_count, "ncclReduceScatter", [comm, out_count](
                                                                     const float* in,
                                                                     float* out,
                                                                     cudaStream_t stream) {
    check_nccl(ncclReduceScatter(in, out, out_count, ncclFloat32, ncclSum, comm, stream), "ncclReduceScatter");
  });
}

std::vector<float> NcclProcessGroup::all_gather(const std::vector<float>& value) const {
  if (value.empty()) return {};
  const std::size_t out_count = value.size() * static_cast<std::size_t>(world_size_);
  auto comm = static_cast<ncclComm_t>(comm_);
  return copy_collective(comm, value, out_count, "ncclAllGather", [comm, count = value.size()](
                                                                    const float* in,
                                                                    float* out,
                                                                    cudaStream_t stream) {
    check_nccl(ncclAllGather(in, out, count, ncclFloat32, comm, stream), "ncclAllGather");
  });
}

std::string nccl_create_unique_id_hex() {
  ncclUniqueId id{};
  check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  return to_hex(id.internal, NCCL_UNIQUE_ID_BYTES);
}

#else

NcclProcessGroup::NcclProcessGroup(int, int, const std::string&) {
  throw std::runtime_error("underHFS native core was built without UNDERHFS_WITH_NCCL=ON");
}

NcclProcessGroup::~NcclProcessGroup() = default;

int NcclProcessGroup::rank() const { return rank_; }

int NcclProcessGroup::world_size() const { return world_size_; }

void NcclProcessGroup::barrier() const {}

std::vector<float> NcclProcessGroup::all_reduce_sum(const std::vector<float>& value) const { return value; }

std::vector<float> NcclProcessGroup::broadcast(const std::vector<float>& value, int) const { return value; }

std::vector<float> NcclProcessGroup::reduce_scatter(const std::vector<float>& values) const { return values; }

std::vector<float> NcclProcessGroup::all_gather(const std::vector<float>& value) const { return value; }

std::string nccl_create_unique_id_hex() {
  throw std::runtime_error("underHFS native core was built without UNDERHFS_WITH_NCCL=ON");
}

#endif

}  // namespace underhfs
