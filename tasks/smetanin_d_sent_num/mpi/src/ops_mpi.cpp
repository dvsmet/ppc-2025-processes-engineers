#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <string>

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
  const InType &text_data = GetInput();

  int process_count = 1;
  int process_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  std::size_t text_length = text_data.length();
  std::size_t chunk_size = text_length / process_count;

  std::size_t segment_start = chunk_size * process_rank;
  std::size_t segment_end = chunk_size + segment_start;

  if (process_rank == process_count - 1) {
    segment_end = text_length;
  }

  unsigned long long local_sentence_count = 0ULL;

  for (std::size_t position = segment_start; position < segment_end; ++position) {
    char current_symbol = text_data[position];

    if (current_symbol != '.' && current_symbol != '!' && current_symbol != '?') {
      continue;
    }

    if (position > 0) {
      char previous_symbol = text_data[position - 1];
      if (previous_symbol == '.' || previous_symbol == '!' || previous_symbol == '?') {
        continue;
      }
    }

    local_sentence_count++;
  }

  unsigned long long global_sentence_count = 0ULL;
  MPI_Reduce(&local_sentence_count, &global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_sentence_count);

  return true;
}

bool SmetaninDSentNumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
