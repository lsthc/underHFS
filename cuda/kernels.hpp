#pragma once

#include <vector>

namespace underhfs {

std::vector<float> cuda_add_f32_host(const std::vector<float>& left,
                                     const std::vector<float>& right);

}  // namespace underhfs
