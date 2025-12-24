#include "../include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../../common/include/common.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

MultiplicationSparseMatricesCRSMPI::MultiplicationSparseMatricesCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRSMatrix{};
}

bool MultiplicationSparseMatricesCRSMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &[a, b] = GetInput();
    return (a.cols == b.rows) && (a.rows > 0) && (b.cols > 0);
  }
  return true;
}

bool MultiplicationSparseMatricesCRSMPI::PreProcessingImpl() {
  return true;
}

bool MultiplicationSparseMatricesCRSMPI::PostProcessingImpl() {
  return true;
}

void MultiplicationSparseMatricesCRSMPI::BroadcastB(int b_nnz, int b_rows, int b_cols, CRSMatrix &b_local) {
  b_local.rows = b_rows;
  b_local.cols = b_cols;
  b_local.nnz = b_nnz;
  b_local.values.resize(b_nnz);
  b_local.col_indices.resize(b_nnz);
  b_local.row_ptr.resize(b_rows + 1);

  MPI_Bcast(b_local.values.data(), b_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_local.col_indices.data(), b_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_local.row_ptr.data(), b_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void MultiplicationSparseMatricesCRSMPI::CalculateSendCountsAndDispls(int a_rows, int size, const CRSMatrix &a_global,
                                                                      std::vector<int> &send_counts,
                                                                      std::vector<int> &displacements) {
  int rows_per_process = a_rows / size;
  int remainder = a_rows % size;

  int current_position = 0;
  for (int process = 0; process < size; ++process) {
    int process_rows = rows_per_process + (process < remainder ? 1 : 0);
    displacements[process] = a_global.row_ptr[current_position];
    send_counts[process] = a_global.row_ptr[current_position + process_rows] - a_global.row_ptr[current_position];
    current_position += process_rows;
  }
}

void MultiplicationSparseMatricesCRSMPI::ScatterRowPtr(int a_rows, int size, int local_rows, const CRSMatrix &a_global,
                                                       std::vector<int> &received_row_nnz) {
  int rows_per_process = a_rows / size;
  int remainder = a_rows % size;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> row_send_counts(size);
  std::vector<int> row_displacements(size, 0);

  if (rank == 0) {
    std::vector<int> all_row_nnz(a_rows);
    for (int i = 0; i < a_rows; ++i) {
      all_row_nnz[i] = a_global.row_ptr[i + 1] - a_global.row_ptr[i];
    }

    for (int process = 0; process < size; ++process) {
      int process_rows = rows_per_process + (process < remainder ? 1 : 0);
      row_send_counts[process] = process_rows;
      if (process > 0) {
        row_displacements[process] = row_displacements[process - 1] + row_send_counts[process - 1];
      }
    }

    MPI_Scatterv(all_row_nnz.data(), row_send_counts.data(), row_displacements.data(), MPI_INT, received_row_nnz.data(),
                 local_rows, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, received_row_nnz.data(), local_rows, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

int MultiplicationSparseMatricesCRSMPI::CalculateProcessStart(int process, int rows_per_process, int remainder) {
  if (process < remainder) {
    return process * (rows_per_process + 1);
  }
  return remainder * (rows_per_process + 1) + (process - remainder) * rows_per_process;
}

int MultiplicationSparseMatricesCRSMPI::CalculateProcessRows(int process, int rows_per_process, int remainder) {
  return rows_per_process + (process < remainder ? 1 : 0);
}

void MultiplicationSparseMatricesCRSMPI::ReceiveProcessData(int process, int process_rows, int &current_nnz,
                                                            CRSMatrix &c, std::vector<int> &process_row_ptr) {
  int process_nnz = 0;
  MPI_Recv(&process_nnz, 1, MPI_INT, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  process_row_ptr.resize(process_rows + 1);
  MPI_Recv(process_row_ptr.data(), process_rows + 1, MPI_INT, process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (process_nnz > 0) {
    std::vector<double> process_values(process_nnz);
    std::vector<int> process_cols(process_nnz);
    MPI_Recv(process_values.data(), process_nnz, MPI_DOUBLE, process, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(process_cols.data(), process_nnz, MPI_INT, process, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    c.values.insert(c.values.end(), process_values.begin(), process_values.end());
    c.col_indices.insert(c.col_indices.end(), process_cols.begin(), process_cols.end());

    current_nnz += process_nnz;
  }
}

void MultiplicationSparseMatricesCRSMPI::UpdateRowPtr(int process_start, int process_rows, int current_nnz,
                                                      const std::vector<int> &process_row_ptr, CRSMatrix &c) {
  for (int i = 0; i < process_rows; ++i) {
    c.row_ptr[process_start + i + 1] = current_nnz + process_row_ptr[i + 1];
  }
}

void MultiplicationSparseMatricesCRSMPI::UpdateRowPtrEmpty(int process_start, int process_rows, int current_nnz,
                                                           CRSMatrix &c) {
  for (int i = 0; i < process_rows; ++i) {
    c.row_ptr[process_start + i + 1] = current_nnz;
  }
}

void MultiplicationSparseMatricesCRSMPI::ComputeLocalC(const CRSMatrix &a_local, const CRSMatrix &b_local,
                                                       CRSMatrix &c_local) {
  c_local.rows = a_local.rows;
  c_local.cols = b_local.cols;
  c_local.row_ptr.resize(a_local.rows + 1);
  c_local.row_ptr[0] = 0;

  std::unordered_map<int, double> accumulator;

  for (int i = 0; i < a_local.rows; ++i) {
    accumulator.clear();

    for (int pa = a_local.row_ptr[i]; pa < a_local.row_ptr[i + 1]; ++pa) {
      int j = a_local.col_indices[pa];
      double val = a_local.values[pa];

      for (int pb = b_local.row_ptr[j]; pb < b_local.row_ptr[j + 1]; ++pb) {
        int k = b_local.col_indices[pb];
        accumulator[k] += val * b_local.values[pb];
      }
    }

    for (const auto &entry : accumulator) {
      if (std::abs(entry.second) > 1e-12) {
        c_local.col_indices.push_back(entry.first);
        c_local.values.push_back(entry.second);
      }
    }

    c_local.row_ptr[i + 1] = static_cast<int>(c_local.values.size());
  }
  c_local.nnz = static_cast<int>(c_local.values.size());
}

void MultiplicationSparseMatricesCRSMPI::DistributeA(int a_rows, CRSMatrix &a_local, int &local_rows, int &local_nnz) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = a_rows / size;
  int remainder = a_rows % size;
  local_rows = rows_per_process + (rank < remainder ? 1 : 0);

  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);

  CRSMatrix a_global;
  if (rank == 0) {
    a_global = std::get<0>(GetInput());
    CalculateSendCountsAndDispls(a_rows, size, a_global, send_counts, displacements);
  }

  MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  a_local.rows = local_rows;
  a_local.cols = std::get<1>(GetInput()).rows;
  a_local.nnz = local_nnz;
  a_local.values.resize(local_nnz);
  a_local.col_indices.resize(local_nnz);
  a_local.row_ptr.resize(local_rows + 1);
  a_local.row_ptr[0] = 0;

  if (local_nnz > 0) {
    MPI_Scatterv(rank == 0 ? a_global.values.data() : nullptr, send_counts.data(), displacements.data(), MPI_DOUBLE,
                 a_local.values.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? a_global.col_indices.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
                 a_local.col_indices.data(), local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }

  std::vector<int> received_row_nnz(local_rows, 0);
  if (local_rows > 0) {
    ScatterRowPtr(a_rows, size, local_rows, a_global, received_row_nnz);
  }

  for (int i = 1; i <= local_rows; ++i) {
    a_local.row_ptr[i] = a_local.row_ptr[i - 1] + received_row_nnz[i - 1];
  }
}

void MultiplicationSparseMatricesCRSMPI::GatherResultOnRoot(int a_rows, int b_cols, int rows_per_process, int remainder,
                                                            const CRSMatrix &c_local, int local_rows, int start_row) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    CRSMatrix c;
    c.rows = a_rows;
    c.cols = b_cols;
    c.row_ptr.resize(a_rows + 1);
    c.row_ptr[0] = 0;

    c.values = c_local.values;
    c.col_indices = c_local.col_indices;

    int current_nnz = c_local.nnz;
    for (int i = 0; i < local_rows; ++i) {
      c.row_ptr[start_row + i + 1] = c_local.row_ptr[i + 1];
    }

    std::vector<int> process_row_ptr;
    for (int process = 1; process < size; ++process) {
      int process_start = CalculateProcessStart(process, rows_per_process, remainder);
      int process_rows = CalculateProcessRows(process, rows_per_process, remainder);

      ReceiveProcessData(process, process_rows, current_nnz, c, process_row_ptr);

      if (!process_row_ptr.empty() && process_row_ptr.back() > 0) {
        UpdateRowPtr(process_start, process_rows, current_nnz - process_row_ptr.back(), process_row_ptr, c);
      } else {
        UpdateRowPtrEmpty(process_start, process_rows, current_nnz, c);
      }
    }

    c.nnz = current_nnz;
    GetOutput() = c;
  } else {
    MPI_Send(&c_local.nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(c_local.row_ptr.data(), local_rows + 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    if (c_local.nnz > 0) {
      MPI_Send(c_local.values.data(), c_local.nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
      MPI_Send(c_local.col_indices.data(), c_local.nnz, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
  }
}

bool MultiplicationSparseMatricesCRSMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  CRSMatrix a_global;
  CRSMatrix b_global;
  int a_rows = 0;
  int b_rows = 0;
  int b_cols = 0;
  int b_nnz = 0;

  if (rank == 0) {
    std::tie(a_global, b_global) = GetInput();
    a_rows = a_global.rows;
    b_rows = b_global.rows;
    b_cols = b_global.cols;
    b_nnz = b_global.nnz;
  }

  MPI_Bcast(&a_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  CRSMatrix b_local;
  if (rank == 0) {
    b_local = b_global;
  }
  BroadcastB(b_nnz, b_rows, b_cols, b_local);

  int local_rows = 0;
  int local_nnz = 0;
  CRSMatrix a_local;
  DistributeA(a_rows, a_local, local_rows, local_nnz);

  CRSMatrix c_local;
  ComputeLocalC(a_local, b_local, c_local);

  int rows_per_process = a_rows / size;
  int remainder = a_rows % size;
  int start_row = rank < remainder ? rank * (rows_per_process + 1)
                                   : remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;

  GatherResultOnRoot(a_rows, b_cols, rows_per_process, remainder, c_local, local_rows, start_row);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
