#include "smetanin_d_sent_num/seq/include/ops_seq.hpp"

#include <cstddef>

#include "smetanin_d_sent_num/common/include/common.hpp"

namespace smetanin_d_sent_num {

SmetaninDSentNumSEQ::SmetaninDSentNumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmetaninDSentNumSEQ::ValidationImpl() {
  const InType &source_data = GetInput();
  const OutType &current_output = GetOutput();
  return !source_data.empty() && current_output == 0;
}

bool SmetaninDSentNumSEQ::PreProcessingImpl() {
  InType &source_text = GetInput();
  if (!source_text.empty()) {
    if (GetInput()[0] == '.' || GetInput()[0] == '!' || GetInput()[0] == '?') {
      GetInput()[0] = ' ';
    }
  }
  return true;
}

bool SmetaninDSentNumSEQ::RunImpl() {
  const InType &text_data = GetInput();
  std::size_t text_length = text_data.length();
  std::size_t sentence_count = 0;

  for (std::size_t current_position = 0; current_position < text_length; ++current_position) {
    char current_char = text_data[current_position];

    if (current_char != '.' && current_char != '!' && current_char != '?') {
      continue;
    }

    if (current_position > 0) {
      char previous_char = text_data[current_position - 1];
      if (previous_char == '.' || previous_char == '!' || previous_char == '?') {
        continue;
      }
    }

    sentence_count++;
  }

  if (sentence_count > 0) {
    GetOutput() = static_cast<OutType>(sentence_count);
  } else {
    GetOutput() = 0;
  }

  return true;
}

bool SmetaninDSentNumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
