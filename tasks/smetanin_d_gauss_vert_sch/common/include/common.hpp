#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace smetanin_d_gauss_vert_sch {

struct GaussBandInput {
  std::vector<double> augmented_matrix;
  int n{0};
  int bandwidth{0};
};

using InType = GaussBandInput;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smetanin_d_gauss_vert_sch
