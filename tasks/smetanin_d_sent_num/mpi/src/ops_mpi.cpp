#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "smetanin_d_sent_num/common/include/common.hpp"

namespace smetanin_d_sent_num {

namespace {
void ComputeSegments(std::size_t text_length, std::size_t proc_count, std::vector<std::size_t> &starts,
                     std::vector<std::size_t> &sizes) {
  const std::size_t base_chunk = text_length / proc_count;
  const std::size_t remainder = text_length % proc_count;
  std::size_t cur = 0;
  for (std::size_t proc = 0; proc < proc_count; ++proc) {
    std::size_t add = (proc < remainder) ? 1U : 0U;
    sizes[proc] = base_chunk + add;
    starts[proc] = cur;
    cur += sizes[proc];
  }
}

void ComputeSendCounts(const std::vector<std::size_t> &starts, const std::vector<std::size_t> &sizes,
                       std::vector<int> &sendcounts, std::vector<int> &displs) {
  const std::size_t proc_count = starts.size();
  for (std::size_t proc = 0; proc < proc_count; ++proc) {
    const std::size_t real_start = starts[proc];
    const std::size_t real_size = sizes[proc];

    if (real_size == 0) {
      sendcounts[proc] = 0;
      displs[proc] = static_cast<int>(real_start);
      continue;
    }

    std::size_t send_start = real_start;
    std::size_t send_size = real_size;
    if (proc != 0 && real_start > 0) {
      send_start = real_start - 1;
      send_size = real_size + 1U;
    }

    sendcounts[proc] = static_cast<int>(send_size);
    displs[proc] = static_cast<int>(send_start);
  }
}

std::size_t CountLocalSentences(const std::string &local_text, int local_start_offset, std::size_t segment_start_global,
                                std::size_t segment_size_global) {
  std::size_t local_sentence_count = 0;
  const int max_index = static_cast<int>(local_text.size());
  const int computed_end = local_start_offset + static_cast<int>(segment_size_global);
  const int end = std::min(max_index, computed_end);
  for (int idx = local_start_offset; idx < end; ++idx) {
    const auto local_idx = static_cast<std::size_t>(idx);
    char current_symbol = local_text[local_idx];

    if (current_symbol != '.' && current_symbol != '!' && current_symbol != '?') {
      continue;
    }

    std::size_t global_pos = segment_start_global + (local_idx - static_cast<std::size_t>(local_start_offset));

    if (global_pos > 0) {
      char previous_symbol = local_text[local_idx - 1];
      if (previous_symbol == '.' || previous_symbol == '!' || previous_symbol == '?') {
        continue;
      }
    }

    local_sentence_count++;
  }
  return local_sentence_count;
}

}  // namespace

SmetaninDSentNumMPI::SmetaninDSentNumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmetaninDSentNumMPI::ValidationImpl() {
  const InType &source_data = GetInput();
  const OutType &current_output = GetOutput();

  return !source_data.empty() && current_output == 0;
}

bool SmetaninDSentNumMPI::PreProcessingImpl() {
  if (GetInput()[0] == '.' || GetInput()[0] == '!' || GetInput()[0] == '?') {
    GetInput()[0] = ' ';
  }
  return true;
}

bool SmetaninDSentNumMPI::RunImpl() {
  int process_count = 1;
  int process_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  std::size_t text_length = 0;
  const InType *full_text_ptr = nullptr;
  if (process_rank == 0) {
    full_text_ptr = &GetInput();
    text_length = full_text_ptr->length();
  }

  auto text_length_u = static_cast<std::uint64_t>(text_length);
  MPI_Bcast(&text_length_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  text_length = static_cast<std::size_t>(text_length_u);

  if (text_length == 0) {
    std::size_t local_sentence_count = 0;
    std::size_t global_sentence_count = 0;

    auto local_u = static_cast<std::uint64_t>(local_sentence_count);
    auto global_u = static_cast<std::uint64_t>(0);
    MPI_Reduce(&local_u, &global_u, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    global_sentence_count = static_cast<std::size_t>(global_u);
    GetOutput() = static_cast<OutType>(global_sentence_count);
    return true;
  }

  const auto proc_count = static_cast<std::size_t>(process_count);
  std::vector<std::size_t> segment_starts(proc_count);
  std::vector<std::size_t> segment_sizes(proc_count);

  ComputeSegments(text_length, proc_count, segment_starts, segment_sizes);

  std::vector<int> sendcounts(proc_count, 0);
  std::vector<int> displs(proc_count, 0);
  ComputeSendCounts(segment_starts, segment_sizes, sendcounts, displs);

  const int local_buffer_size = sendcounts[process_rank];
  std::string local_text(static_cast<std::size_t>(local_buffer_size), ' ');

  const char *sendbuf = nullptr;
  if (process_rank == 0 && full_text_ptr != nullptr) {
    sendbuf = full_text_ptr->data();
  }

  char *recvbuf = local_buffer_size > 0 ? local_text.data() : nullptr;

  MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_CHAR, recvbuf, local_buffer_size, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  std::size_t local_sentence_count = 0;

  if (local_buffer_size > 0) {
    const auto segment_start_global = segment_starts[process_rank];
    const auto segment_size_global = segment_sizes[process_rank];

    const int local_start_offset = (process_rank == 0 || segment_start_global == 0 || segment_size_global == 0) ? 0 : 1;

    local_sentence_count =
        CountLocalSentences(local_text, local_start_offset, segment_start_global, segment_size_global);
  }

  std::size_t global_sentence_count = 0;
  auto local_u = static_cast<std::uint64_t>(local_sentence_count);
  auto global_u = static_cast<std::uint64_t>(0);
  MPI_Reduce(&local_u, &global_u, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  global_sentence_count = static_cast<std::size_t>(global_u);
  GetOutput() = static_cast<OutType>(global_sentence_count);

  return true;
}

bool SmetaninDSentNumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
