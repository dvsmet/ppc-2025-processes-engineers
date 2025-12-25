#pragma once

#include <random>

#include "../common/include/common.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

inline CRSMatrix GenerateRandomCRS(int rows, int cols, double density, int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  CRSMatrix mat;
  mat.rows = rows;
  mat.cols = cols;
  mat.row_ptr.push_back(0);

  int nnz = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (prob_dist(gen) < density) {
        mat.values.push_back(val_dist(gen));
        mat.col_indices.push_back(j);
        ++nnz;
      }
    }
    mat.row_ptr.push_back(nnz);
  }
  mat.nnz = nnz;
  return mat;
}

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
