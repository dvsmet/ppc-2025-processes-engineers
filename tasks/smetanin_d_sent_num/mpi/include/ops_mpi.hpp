#pragma once

#include "smetanin_d_sent_num/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smetanin_d_sent_num {

class SmetaninDSentNumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SmetaninDSentNumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smetanin_d_sent_num
