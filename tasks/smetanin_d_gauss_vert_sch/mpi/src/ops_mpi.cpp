#include "smetanin_d_gauss_vert_sch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "smetanin_d_gauss_vert_sch/common/include/common.hpp"

namespace smetanin_d_gauss_vert_sch {

namespace {

constexpr auto kLocalAt = [](std::vector<double> &data, int n, int row, int lc) -> double & {
  return data[(static_cast<std::size_t>(lc) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(row)];
};

constexpr auto kConstLocalAt = [](const std::vector<double> &data, int n, int row, int lc) -> const double & {
  return data[(static_cast<std::size_t>(lc) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(row)];
};

constexpr auto kGetOwner = [](const std::vector<int> &displs, const std::vector<int> &col_counts, int col) -> int {
  for (size_t pr = 0; pr < col_counts.size(); ++pr) {
    if (displs[pr] <= col && col < displs[pr] + col_counts[pr]) {
      return static_cast<int>(pr);
    }
  }
  return -1;
};

void DistributeColumns(int size, int total_cols, std::vector<int> &col_counts, std::vector<int> &displs) {
  const int base = total_cols / size;
  const int rem = total_cols % size;
  col_counts.resize(size);
  displs.resize(size, 0);

  for (int pr = 0; pr < size; ++pr) {
    col_counts[pr] = base + (pr < rem ? 1 : 0);
    if (pr > 0) {
      displs[pr] = displs[pr - 1] + col_counts[pr - 1];
    }
  }
}

void ScatterMatrix(const InType &input, int n, int start_col, int local_cols, std::vector<double> &local_matrix) {
  local_matrix.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(local_cols));

  for (int lc = 0; lc < local_cols; ++lc) {
    const int gc = start_col + lc;
    for (int row = 0; row < n; ++row) {
      kLocalAt(local_matrix, n, row, lc) =
          input.augmented_matrix[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) +
                                 static_cast<std::size_t>(gc)];
    }
  }
}

bool FindPivotAndSwap(int i, int n, int bw, int rank, int pivot_owner, int start_col, std::vector<double> &local_matrix,
                      double eps) {
  int max_row_local = i;
  double max_val_local = 0.0;

  if (rank == pivot_owner) {
    const int lc = i - start_col;
    max_val_local = std::abs(kConstLocalAt(local_matrix, n, i, lc));
    const int pivot_row_end = std::min(n - 1, i + bw);
    for (int ri = i + 1; ri <= pivot_row_end; ++ri) {
      const double val = std::abs(kConstLocalAt(local_matrix, n, ri, lc));
      if (val > max_val_local) {
        max_val_local = val;
        max_row_local = ri;
      }
    }
  }

  MPI_Bcast(&max_row_local, 1, MPI_INT, pivot_owner, MPI_COMM_WORLD);
  MPI_Bcast(&max_val_local, 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

  if (max_val_local <= eps) {
    return false;
  }

  if (max_row_local != i) {
    const int local_cols = static_cast<int>(local_matrix.size() / n);
    for (int lc = 0; lc < local_cols; ++lc) {
      std::swap(kLocalAt(local_matrix, n, i, lc), kLocalAt(local_matrix, n, max_row_local, lc));
    }
  }

  return true;
}

void ComputeAndBroadcastFactors(int i, int n, int bw, int rank, int pivot_owner, int start_col,
                                std::vector<double> &local_matrix, std::vector<double> &factors) {
  const int elim_end = std::min(n - 1, i + bw);
  const int num_factors = elim_end - i;
  factors.assign(static_cast<std::size_t>(num_factors), 0.0);

  if (rank == pivot_owner) {
    const int lc = i - start_col;
    const double pivot = kConstLocalAt(local_matrix, n, i, lc);
    for (int k = 0; k < num_factors; ++k) {
      const int ri = i + 1 + k;
      double &entry = kLocalAt(local_matrix, n, ri, lc);
      const double factor = entry / pivot;
      factors[static_cast<std::size_t>(k)] = factor;
      entry = 0.0;
    }
  }

  MPI_Bcast(factors.data(), num_factors, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
}

void EliminateBelowPivot(int i, int n, int bw, int start_col, int local_cols, const std::vector<double> &factors,
                         std::vector<double> &local_matrix, double eps) {
  const int num_factors = static_cast<int>(factors.size());
  for (int k = 0; k < num_factors; ++k) {
    const int ri = i + 1 + k;
    const double factor = factors[static_cast<std::size_t>(k)];
    if (std::abs(factor) <= eps) {
      continue;
    }
    for (int lc = 0; lc < local_cols; ++lc) {
      const int gc = start_col + lc;
      if (gc <= i) {
        continue;
      }
      if (gc < n && gc > i + bw) {
        continue;
      }
      kLocalAt(local_matrix, n, ri, lc) -= factor * kConstLocalAt(local_matrix, n, i, lc);
    }
  }
}

void BackSubstitution(int n, int bw, int rank, int b_owner, int start_col, int local_cols,
                      const std::vector<double> &local_matrix, const std::vector<int> &displs,
                      const std::vector<int> &col_counts, OutType &sol, double eps) {
  const int b_col = n;

  for (int i = n - 1; i >= 0; --i) {
    const int diag_owner = kGetOwner(displs, col_counts, i);

    double local_sum = 0.0;
    const int col_end = std::min(n - 1, i + bw);
    for (int lc = 0; lc < local_cols; ++lc) {
      const int gc = start_col + lc;
      if (gc <= i || gc > col_end) {
        continue;
      }
      local_sum += kConstLocalAt(local_matrix, n, i, lc) * sol[static_cast<std::size_t>(gc)];
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_val = 0.0;
    if (rank == b_owner) {
      const int lc_b = b_col - start_col;
      b_val = kConstLocalAt(local_matrix, n, i, lc_b);
    }
    MPI_Bcast(&b_val, 1, MPI_DOUBLE, b_owner, MPI_COMM_WORLD);

    double diag = 0.0;
    if (rank == diag_owner) {
      const int lc_d = i - start_col;
      diag = kConstLocalAt(local_matrix, n, i, lc_d);
    }
    MPI_Bcast(&diag, 1, MPI_DOUBLE, diag_owner, MPI_COMM_WORLD);

    if (std::abs(diag) <= eps) {
      sol.clear();
      return;
    }

    const double sum = b_val - global_sum;
    sol[static_cast<std::size_t>(i)] = sum / diag;
  }
}

}  // anonymous namespace

SmetaninDGaussVertSchMPI::SmetaninDGaussVertSchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SmetaninDGaussVertSchMPI::ValidationImpl() {
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

bool SmetaninDGaussVertSchMPI::PreProcessingImpl() {
  return true;
}

bool SmetaninDGaussVertSchMPI::RunImpl() {
  const InType &input = GetInput();
  const int n = input.n;
  const int bw = input.bandwidth;
  const double eps = std::numeric_limits<double>::epsilon() * 100.0;

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_cols = n + 1;

  std::vector<int> col_counts;
  std::vector<int> displs;
  DistributeColumns(size, total_cols, col_counts, displs);

  const int start_col = displs[rank];
  const int local_cols = col_counts[rank];

  std::vector<double> local_matrix;
  ScatterMatrix(input, n, start_col, local_cols, local_matrix);

  for (int i = 0; i < n; ++i) {
    const int pivot_owner = kGetOwner(displs, col_counts, i);

    if (!FindPivotAndSwap(i, n, bw, rank, pivot_owner, start_col, local_matrix, eps)) {
      return false;
    }

    std::vector<double> factors;
    ComputeAndBroadcastFactors(i, n, bw, rank, pivot_owner, start_col, local_matrix, factors);

    EliminateBelowPivot(i, n, bw, start_col, local_cols, factors, local_matrix, eps);
  }

  OutType sol(static_cast<std::size_t>(n), 0.0);
  const int b_owner = kGetOwner(displs, col_counts, n);

  BackSubstitution(n, bw, rank, b_owner, start_col, local_cols, local_matrix, displs, col_counts, sol, eps);

  if (sol.empty()) {
    return false;
  }

  GetOutput() = std::move(sol);
  return true;
}

bool SmetaninDGaussVertSchMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace smetanin_d_gauss_vert_sch
