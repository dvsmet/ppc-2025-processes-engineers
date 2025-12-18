#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "smetanin_d_sent_num/common/include/common.hpp"
#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"
#include "smetanin_d_sent_num/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace smetanin_d_sent_num {

class SmetaninDRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType task_answer_ = 0;

  void SetUp() override {
    std::string test_file_path = "test_5.txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_smetanin_d_sent_num, test_file_path);

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    std::stringstream ss;
    ss << file.rdbuf();
    input_data_ = ss.str();
    file.close();

    task_answer_ = 0;
    for (size_t i = 0; i < input_data_.length(); ++i) {
      char c = input_data_[i];
      if (c == '.' || c == '!' || c == '?') {
        if (i == 0 || (input_data_[i - 1] != '.' && input_data_[i - 1] != '!' && input_data_[i - 1] != '?')) {
          task_answer_++;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == task_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmetaninDRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SmetaninDSentNumMPI, SmetaninDSentNumSEQ>(PPC_SETTINGS_smetanin_d_sent_num);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmetaninDRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmetaninDRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace smetanin_d_sent_num
