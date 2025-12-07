#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
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
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  (void)process_rank;
  return true;
}

bool SmetaninDSentNumMPI::RunImpl() {
  const InType &text_data = GetInput();

  int process_count = 1;
  int process_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  std::size_t text_length = 0;
  if (process_rank == 0) {
    text_length = text_data.length();
  }
  auto text_length_u64 = static_cast<uint64_t>(text_length);
  MPI_Bcast(&text_length_u64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  text_length = static_cast<std::size_t>(text_length_u64);

  std::vector<int> sendcounts(process_count, 0);
  std::vector<int> displs(process_count, 0);
  if (process_rank == 0) {
    std::size_t base = text_length / static_cast<std::size_t>(process_count);
    std::size_t remainder = text_length % static_cast<std::size_t>(process_count);
    for (int i = 0; i < process_count; ++i) {
      std::size_t cnt = base + (static_cast<std::size_t>(i) < remainder ? 1U : 0U);
      sendcounts[i] = static_cast<int>(cnt);
      if (i > 0) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
      }
    }
  }

  int local_count = 0;
  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_count < 0) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }

  std::vector<char> local_buffer;
  if (local_count > 0) {
    local_buffer.resize(local_count);
  }

  const char *sendbuf = nullptr;
  if (process_rank == 0 && text_length > 0) {
    sendbuf = text_data.data();
  }

  char dummy_recv = '\0';
  char *recv_ptr = local_count > 0 ? local_buffer.data() : &dummy_recv;

  int scatterv_ret = MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_CHAR, recv_ptr, local_count, MPI_CHAR,
                                  0, MPI_COMM_WORLD);
  if (scatterv_ret != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, scatterv_ret);
    return false;
  }

  std::string local_chunk;
  if (local_count > 0) {
    local_chunk.assign(local_buffer.begin(), local_buffer.end());
  } else {
    local_chunk.clear();
  }

  auto is_term = [](char ch) { return ch == '.' || ch == '!' || ch == '?'; };

  char prev_char = '\0';
  std::vector<char> prev_chars;
  if (process_rank == 0) {
    prev_chars.resize(process_count);
    for (int i = 0; i < process_count; ++i) {
      if (displs[i] == 0) {
        prev_chars[i] = '\0';
      } else {
        prev_chars[i] = text_data[static_cast<std::size_t>(displs[i]) - 1];
      }
    }
  }
  MPI_Scatter(prev_chars.data(), 1, MPI_CHAR, &prev_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  uint64_t local_sentence_count = 0ULL;
  for (std::size_t pos = 0; pos < local_chunk.size(); ++pos) {
    char c = local_chunk[pos];
    if (!is_term(c)) {
      continue;
    }
    if (pos == 0) {
      if (prev_char != '\0' && !is_term(prev_char)) {
        ++local_sentence_count;
      }
    } else {
      if (!is_term(local_chunk[pos - 1])) {
        ++local_sentence_count;
      }
    }
    while (pos + 1 < local_chunk.size() && is_term(local_chunk[pos + 1])) {
      ++pos;
    }
  }

  uint64_t global_sentence_count = 0ULL;
  MPI_Reduce(&local_sentence_count, &global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_sentence_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  GetOutput() = static_cast<OutType>(global_sentence_count);

  return true;
}

bool SmetaninDSentNumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
