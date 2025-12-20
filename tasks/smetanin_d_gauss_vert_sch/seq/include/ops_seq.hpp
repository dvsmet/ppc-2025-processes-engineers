#pragma once

#include "smetanin_d_gauss_vert_sch/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smetanin_d_gauss_vert_sch {

class SmetaninDGaussVertSchSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SmetaninDGaussVertSchSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smetanin_d_gauss_vert_sch
