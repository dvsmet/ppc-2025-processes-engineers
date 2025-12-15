#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "smetanin_d_sent_num/common/include/common.hpp"

namespace smetanin_d_sent_num {

namespace {
unsigned long long ToMpiUll(std::uint64_t value) {
  return static_cast<unsigned long long>(value);
}

std::uint64_t FromMpiUll(unsigned long long value) {
  return static_cast<std::uint64_t>(value);
}

std::uint64_t BroadcastTextLength(std::size_t &text_length, const std::string &text_data, int process_rank) {
  if (process_rank == 0) {
    text_length = text_data.length();
  }
  auto text_length_uint64 = static_cast<std::uint64_t>(text_length);
  auto text_length_ull = ToMpiUll(text_length_uint64);
  MPI_Bcast(&text_length_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  text_length_uint64 = FromMpiUll(text_length_ull);
  text_length = static_cast<std::size_t>(text_length_uint64);
  return text_length_uint64;
}

void ComputeSendCountsAndDispls(std::size_t text_length, int process_count, std::vector<int> &sendcounts,
                                std::vector<int> &displs) {
  auto pcount = static_cast<std::size_t>(process_count);
  std::size_t base = text_length / pcount;
  std::size_t remainder = text_length % pcount;
  sendcounts.assign(pcount, 0);
  displs.assign(pcount, 0);
  for (std::size_t i = 0; i < pcount; ++i) {
    std::size_t cnt = base + (i < remainder ? 1U : 0U);
    sendcounts[static_cast<int>(i)] = static_cast<int>(cnt);
    if (i > 0) {
      displs[static_cast<int>(i)] = displs[static_cast<int>(i) - 1] + sendcounts[static_cast<int>(i) - 1];
    }
  }
}

bool IsTerminalChar(char ch) {
  return ch == '.' || ch == '!' || ch == '?';
}

std::uint64_t CountSentencesInChunk(const std::string &local_chunk, char prev_char) {
  std::uint64_t local_sentence_count = 0ULL;
  for (std::size_t pos = 0; pos < local_chunk.size(); ++pos) {
    char c = local_chunk[pos];
    if (!IsTerminalChar(c)) {
      continue;
    }
    if (pos == 0) {
      if (prev_char != '\0' && !IsTerminalChar(prev_char)) {
        ++local_sentence_count;
      }
    } else {
      if (!IsTerminalChar(local_chunk[pos - 1])) {
        ++local_sentence_count;
      }
    }
    while (pos + 1 < local_chunk.size() && IsTerminalChar(local_chunk[pos + 1])) {
      ++pos;
    }
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
  BroadcastTextLength(text_length, text_data, process_rank);

  std::vector<int> sendcounts;
  std::vector<int> displs;
  if (process_rank == 0) {
    ComputeSendCountsAndDispls(text_length, process_count, sendcounts, displs);
  } else {
    sendcounts.assign(static_cast<std::size_t>(process_count), 0);
    displs.assign(static_cast<std::size_t>(process_count), 0);
  }

  int local_count = 0;
  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_count < 0) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }

  std::vector<char> local_buffer;
  if (local_count > 0) {
    local_buffer.resize(static_cast<std::size_t>(local_count));
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

  char prev_char = '\0';
  std::vector<char> prev_chars;
  if (process_rank == 0) {
    prev_chars.resize(static_cast<std::size_t>(process_count));
    for (int i = 0; i < process_count; ++i) {
      if (displs[i] == 0) {
        prev_chars[i] = '\0';
      } else {
        prev_chars[i] = text_data[static_cast<std::size_t>(displs[i]) - 1];
      }
    }
  }
  MPI_Scatter(prev_chars.data(), 1, MPI_CHAR, &prev_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  std::uint64_t local_sentence_count = CountSentencesInChunk(local_chunk, prev_char);

  std::uint64_t global_sentence_count = 0ULL;
  auto local_sentence_count_ull = ToMpiUll(local_sentence_count);
  auto global_sentence_count_ull = ToMpiUll(0ULL);
  MPI_Reduce(&local_sentence_count_ull, &global_sentence_count_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  MPI_Bcast(&global_sentence_count_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  global_sentence_count = FromMpiUll(global_sentence_count_ull);
  GetOutput() = static_cast<OutType>(global_sentence_count);

  return true;
}

bool SmetaninDSentNumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
