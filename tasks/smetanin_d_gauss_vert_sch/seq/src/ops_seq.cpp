#include "smetanin_d_gauss_vert_sch/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "smetanin_d_gauss_vert_sch/common/include/common.hpp"

namespace smetanin_d_gauss_vert_sch {

namespace {

double &At(std::vector<double> &data, int n, int row, int col) {
  return data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}

const double &At(const std::vector<double> &data, int n, int row, int col) {
  return data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}

bool FindAndSwapPivot(std::vector<double> &a, int n, int i, int bw, double eps) {
  int max_row = i;
  double max_val = std::abs(At(a, n, i, i));
  const int pivot_row_end = std::min(n - 1, i + bw);
  for (int ri = i + 1; ri <= pivot_row_end; ++ri) {
    const double val = std::abs(At(a, n, ri, i));
    if (val > max_val) {
      max_val = val;
      max_row = ri;
    }
  }
  if (max_val <= eps) {
    return false;
  }
  if (max_row != i) {
    const int col_swap_end = std::min(n - 1, i + bw);
    for (int cj = i; cj <= col_swap_end; ++cj) {
      std::swap(At(a, n, i, cj), At(a, n, max_row, cj));
    }
    std::swap(At(a, n, i, n), At(a, n, max_row, n));
  }
  return true;
}

void EliminateBelow(std::vector<double> &a, int n, int i, int bw, double eps) {
  for (int ri = i + 1; ri <= std::min(n - 1, i + bw); ++ri) {
    const double factor = At(a, n, ri, i) / At(a, n, i, i);
    if (std::abs(factor) <= eps) {
      continue;
    }
    At(a, n, ri, i) = 0.0;
    const int col_end = std::min(n - 1, i + bw);
    for (int cj = i + 1; cj <= col_end; ++cj) {
      At(a, n, ri, cj) -= factor * At(a, n, i, cj);
    }
    At(a, n, ri, n) -= factor * At(a, n, i, n);
  }
}

OutType BackSubstitute(const std::vector<double> &a, int n, int bw, double eps) {
  OutType x(static_cast<std::size_t>(n), 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double sum = At(a, n, i, n);
    const int col_end = std::min(n - 1, i + bw);
    for (int cj = i + 1; cj <= col_end; ++cj) {
      sum -= At(a, n, i, cj) * x[static_cast<std::size_t>(cj)];
    }
    const double diag = At(a, n, i, i);
    if (std::abs(diag) <= eps) {
      return OutType{};
    }
    x[static_cast<std::size_t>(i)] = sum / diag;
  }
  return x;
}

}  // namespace

SmetaninDGaussVertSchSEQ::SmetaninDGaussVertSchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SmetaninDGaussVertSchSEQ::ValidationImpl() {
  const InType &input = GetInput();
  const OutType &output = GetOutput();

  if (input.n <= 0 || input.bandwidth < 0 || input.bandwidth >= input.n) {
    return false;
  }

  const std::size_t expected_size = static_cast<std::size_t>(input.n) * static_cast<std::size_t>(input.n + 1);
  if (input.augmented_matrix.size() != expected_size) {
    return false;
  }

  return output.empty();
}

bool SmetaninDGaussVertSchSEQ::PreProcessingImpl() {
  return true;
}

bool SmetaninDGaussVertSchSEQ::RunImpl() {
  const InType &input = GetInput();
  const int n = input.n;
  const int bw = input.bandwidth;

  std::vector<double> a = input.augmented_matrix;

  const double eps = std::numeric_limits<double>::epsilon() * 100.0;

  for (int i = 0; i < n; ++i) {
    if (!FindAndSwapPivot(a, n, i, bw, eps)) {
      return false;
    }
    EliminateBelow(a, n, i, bw, eps);
  }

  OutType x = BackSubstitute(a, n, bw, eps);
  if (x.empty()) {
    return false;
  }
  GetOutput() = std::move(x);
  return true;
}

bool SmetaninDGaussVertSchSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace smetanin_d_gauss_vert_sch
