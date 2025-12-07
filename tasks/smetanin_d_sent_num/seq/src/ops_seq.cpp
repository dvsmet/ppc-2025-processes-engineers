#include "smetanin_d_sent_num/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

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
  return true;
}

bool SmetaninDSentNumSEQ::RunImpl() {
  const InType &text = GetInput();

  auto is_term = [](char ch) { return ch == '.' || ch == '!' || ch == '?'; };

  std::size_t sentence_count = 0;
  bool seen_text = false;

  for (std::size_t i = 0; i < text.size();) {
    char c = text[i];
    if (!is_term(c)) {
      seen_text = true;
      ++i;
      continue;
    }

    if (seen_text) {
      ++sentence_count;
    }
    while (i < text.size() && is_term(text[i])) {
      ++i;
    }
    seen_text = false;
  }

  GetOutput() = static_cast<OutType>(sentence_count);
  return true;
}

bool SmetaninDSentNumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smetanin_d_sent_num
