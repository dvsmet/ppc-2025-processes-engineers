#pragma once

#include <vector>

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

class MultiplicationSparseMatricesCRSMPI : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MultiplicationSparseMatricesCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void BroadcastB(int b_nnz, int b_rows, int b_cols, CRSMatrix &b_local);

  static void CalculateSendCountsAndDispls(int a_rows, int size, const CRSMatrix &a_global,
                                           std::vector<int> &send_counts, std::vector<int> &displacements);

  static void ScatterRowPtr(int a_rows, int size, int local_rows, const CRSMatrix &a_global,
                            std::vector<int> &received_row_nnz);

  static int CalculateProcessStart(int process, int rows_per_process, int remainder);

  static int CalculateProcessRows(int process, int rows_per_process, int remainder);

  static void ReceiveProcessData(int process, int process_rows, int &current_nnz, CRSMatrix &c,
                                 std::vector<int> &process_row_ptr);

  static void UpdateRowPtr(int process_start, int process_rows, int current_nnz,
                           const std::vector<int> &process_row_ptr, CRSMatrix &c);

  static void UpdateRowPtrEmpty(int process_start, int process_rows, int current_nnz, CRSMatrix &c);

  static void ComputeLocalC(const CRSMatrix &a_local, const CRSMatrix &b_local, CRSMatrix &c_local);

  void DistributeA(int a_rows, CRSMatrix &a_local, int &local_rows, int &local_nnz);

  void GatherResultOnRoot(int a_rows, int b_cols, int rows_per_process, int remainder, const CRSMatrix &c_local,
                          int local_rows, int start_row);
};

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
