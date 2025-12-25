#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

class MultiplicationSparseMatricesCRSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MultiplicationSparseMatricesCRSSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
