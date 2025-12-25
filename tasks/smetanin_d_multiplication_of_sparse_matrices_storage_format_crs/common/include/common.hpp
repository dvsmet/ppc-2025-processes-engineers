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

inline CRSMatrix MultiplyCRS(const CRSMatrix &a_matrix, const CRSMatrix &b_matrix) {
  if (a_matrix.cols != b_matrix.rows) {
    throw std::invalid_argument("Incompatible matrix dimensions");
  }

  CRSMatrix c_matrix;
  c_matrix.rows = a_matrix.rows;
  c_matrix.cols = b_matrix.cols;
  c_matrix.row_ptr.resize(a_matrix.rows + 1);
  c_matrix.row_ptr[0] = 0;

  std::unordered_map<int, double> accumulator;

  for (int i = 0; i < a_matrix.rows; ++i) {
    accumulator.clear();

    for (int pa = a_matrix.row_ptr[i]; pa < a_matrix.row_ptr[i + 1]; ++pa) {
      int j = a_matrix.col_indices[pa];
      double val = a_matrix.values[pa];

      for (int pb = b_matrix.row_ptr[j]; pb < b_matrix.row_ptr[j + 1]; ++pb) {
        int k = b_matrix.col_indices[pb];
        accumulator[k] += val * b_matrix.values[pb];
      }
    }

    for (const auto &entry : accumulator) {
      if (std::abs(entry.second) > 1e-12) {
        c_matrix.col_indices.push_back(entry.first);
        c_matrix.values.push_back(entry.second);
      }
    }

    c_matrix.row_ptr[i + 1] = static_cast<int>(c_matrix.values.size());
  }

  c_matrix.nnz = static_cast<int>(c_matrix.values.size());
  return c_matrix;
}

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
