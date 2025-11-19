#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace smetanin_d_sent_num {

using InType = std::string;
using OutType = std::size_t;
using TestType = std::tuple<int, std::string, std::size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smetanin_d_sent_num
