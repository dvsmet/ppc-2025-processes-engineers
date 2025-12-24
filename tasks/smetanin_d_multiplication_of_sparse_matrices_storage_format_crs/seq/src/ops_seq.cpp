#include "../include/ops_seq.hpp"

#include "../../common/include/common.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

MultiplicationSparseMatricesCRSSEQ::MultiplicationSparseMatricesCRSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRSMatrix{};
}

bool MultiplicationSparseMatricesCRSSEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return (a.cols == b.rows) && (a.rows > 0) && (b.cols > 0);
}

bool MultiplicationSparseMatricesCRSSEQ::PreProcessingImpl() {
  return true;
}

bool MultiplicationSparseMatricesCRSSEQ::RunImpl() {
  const auto &[a, b] = GetInput();
  GetOutput() = MultiplyCRS(a, b);
  return true;
}

bool MultiplicationSparseMatricesCRSSEQ::PostProcessingImpl() {
  const auto &[a, b] = GetInput();
  return GetOutput().nnz > 0 || (a.rows == 0 || b.cols == 0);
}

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
