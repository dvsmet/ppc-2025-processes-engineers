#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "task/include/task.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

struct CRSMatrix {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;
};

using InType = std::tuple<CRSMatrix, CRSMatrix>;
using OutType = CRSMatrix;
using TestType = std::tuple<int, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline CRSMatrix MultiplyCRS(const CRSMatrix &A, const CRSMatrix &B) {
  if (A.cols != B.rows) {
    throw std::invalid_argument("Incompatible matrix dimensions");
  }

  CRSMatrix C;
  C.rows = A.rows;
  C.cols = B.cols;
  C.row_ptr.resize(A.rows + 1);
  C.row_ptr[0] = 0;

  std::unordered_map<int, double> accumulator;

  for (int i = 0; i < A.rows; ++i) {
    accumulator.clear();

    for (int pa = A.row_ptr[i]; pa < A.row_ptr[i + 1]; ++pa) {
      int j = A.col_indices[pa];
      double val = A.values[pa];

      for (int pb = B.row_ptr[j]; pb < B.row_ptr[j + 1]; ++pb) {
        int k = B.col_indices[pb];
        accumulator[k] += val * B.values[pb];
      }
    }

    for (const auto &[k, v] : accumulator) {
      if (std::abs(v) > 1e-12) {
        C.col_indices.push_back(k);
        C.values.push_back(v);
      }
    }

    C.row_ptr[i + 1] = static_cast<int>(C.values.size());
  }

  C.nnz = static_cast<int>(C.values.size());
  return C;
}

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
