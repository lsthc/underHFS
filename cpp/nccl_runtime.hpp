#pragma once

#include <string>
#include <vector>

namespace underhfs {

class NcclProcessGroup {
 public:
  NcclProcessGroup(int rank, int world_size, const std::string& unique_id_hex = "");
  ~NcclProcessGroup();

  NcclProcessGroup(const NcclProcessGroup&) = delete;
  NcclProcessGroup& operator=(const NcclProcessGroup&) = delete;

  int rank() const;
  int world_size() const;
  void barrier() const;
  std::vector<float> all_reduce_sum(const std::vector<float>& value) const;
  std::vector<float> broadcast(const std::vector<float>& value, int src) const;
  std::vector<float> reduce_scatter(const std::vector<float>& values) const;
  std::vector<float> all_gather(const std::vector<float>& value) const;

 private:
  int rank_ = 0;
  int world_size_ = 1;
  void* comm_ = nullptr;
};

std::string nccl_create_unique_id_hex();

}  // namespace underhfs
