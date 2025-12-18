#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "smetanin_d_sent_num/common/include/common.hpp"

namespace smetanin_d_sent_num {

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

  MPI_Bcast(&text_length, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (text_length == 0) {
    std::size_t local_sentence_count = 0;
    std::size_t global_sentence_count = 0;

    MPI_Reduce(&local_sentence_count, &global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    GetOutput() = static_cast<OutType>(global_sentence_count);
    return true;
  }

  std::vector<std::size_t> segment_starts(process_count);
  std::vector<std::size_t> segment_sizes(process_count);

  const std::size_t base_chunk = text_length / static_cast<std::size_t>(process_count);
  const std::size_t remainder = text_length % static_cast<std::size_t>(process_count);

  std::size_t current_start = 0;
  for (int r = 0; r < process_count; ++r) {
    std::size_t size = base_chunk + (static_cast<std::size_t>(r) < remainder ? 1u : 0u);
    segment_starts[r] = current_start;
    segment_sizes[r] = size;
    current_start += size;
  }

  std::vector<int> sendcounts(process_count, 0);
  std::vector<int> displs(process_count, 0);

  for (int r = 0; r < process_count; ++r) {
    std::size_t real_start = segment_starts[r];
    std::size_t real_size = segment_sizes[r];

    if (real_size == 0) {
      sendcounts[r] = 0;
      displs[r] = static_cast<int>(real_start);
      continue;
    }

    std::size_t send_start = real_start;
    std::size_t send_size = real_size;

    if (r != 0 && real_start > 0) {
      send_start = real_start - 1;
      send_size = real_size + 1;
    }

    sendcounts[r] = static_cast<int>(send_size);
    displs[r] = static_cast<int>(send_start);
  }

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
    const std::size_t segment_start_global = segment_starts[process_rank];
    const std::size_t segment_size_global = segment_sizes[process_rank];

    const int local_start_offset = (process_rank == 0 || segment_start_global == 0 || segment_size_global == 0) ? 0 : 1;

    for (int i = local_start_offset; i < local_buffer_size; ++i) {
      char current_symbol = local_text[static_cast<std::size_t>(i)];

      if (current_symbol != '.' && current_symbol != '!' && current_symbol != '?') {
        continue;
      }

      std::size_t global_pos = segment_start_global + static_cast<std::size_t>(i - local_start_offset);

      if (global_pos > 0) {
        char previous_symbol = local_text[static_cast<std::size_t>(i) - 1];
        if (previous_symbol == '.' || previous_symbol == '!' || previous_symbol == '?') {
          continue;
        }
      }

      local_sentence_count++;
    }
  }

  std::size_t global_sentence_count = 0;

  MPI_Reduce(&local_sentence_count, &global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_sentence_count);

  return true;
}

bool SmetaninDSentNumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
