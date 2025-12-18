#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "smetanin_d_sent_num/common/include/common.hpp"
#include "smetanin_d_sent_num/mpi/include/ops_mpi.hpp"
#include "smetanin_d_sent_num/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace smetanin_d_sent_num {

class SmetaninDSentNumTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_index = std::get<0>(test_param);
    const std::string &file_identifier = std::get<1>(test_param);

    if (file_identifier.empty()) {
      return std::to_string(test_index) + "_generated";
    }
    return std::to_string(test_index) + "_from_file";
  }

 protected:
  void SetUp() override {
    std::string text;
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_file_path = std::get<1>(params);
    if (test_file_path.empty()) {
      task_answer_ = std::get<2>(params);
      input_data_ = GenerateTestData(task_answer_, 0);
    } else {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_smetanin_d_sent_num, test_file_path);
      std::ifstream file(abs_path);
      if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + abs_path);
      }
      std::stringstream ss;
      ss << file.rdbuf();
      text = ss.str();

      task_answer_ = std::get<2>(params);
      input_data_ = text;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == task_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType task_answer_ = 0;

  static std::string LoadContentFromFile(const std::string &filepath) {
    std::string full_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_smetanin_d_sent_num, filepath);

    std::ifstream input_file(full_path);
    if (!input_file.is_open()) {
      throw std::runtime_error("Cannot open test file: " + full_path);
    }

    std::stringstream file_buffer;
    file_buffer << input_file.rdbuf();
    input_file.close();

    return file_buffer.str();
  }

  static std::string GenerateTestData(const std::size_t sentence_count, const int random_seed) {
    std::mt19937 random_generator(random_seed);
    std::uniform_int_distribution<> character_distribution('A', 'z');

    std::string generated_text;
    generated_text.reserve(sentence_count * 20);

    for (std::size_t sentence_index = 0; sentence_index < sentence_count; ++sentence_index) {
      int generated_length = (character_distribution(random_generator) % 50) + 5;

      for (int char_position = 0; char_position < generated_length; ++char_position) {
        char random_char = static_cast<char>(character_distribution(random_generator));
        generated_text += random_char;
      }

      generated_text += '.';
    }

    return generated_text;
  }
};

namespace {

TEST_P(SmetaninDSentNumTestsProcesses, SentenceCountFromText) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(0, "test_0.txt", 0), std::make_tuple(1, "test_1.txt", 1),
                                            std::make_tuple(2, "test_2.txt", 4), std::make_tuple(3, "test_3.txt", 100),
                                            std::make_tuple(4, "test_4.txt", 1)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SmetaninDSentNumMPI, InType>(kTestParam, PPC_SETTINGS_smetanin_d_sent_num),
                   ppc::util::AddFuncTask<SmetaninDSentNumSEQ, InType>(kTestParam, PPC_SETTINGS_smetanin_d_sent_num));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SmetaninDSentNumTestsProcesses::PrintFuncTestName<SmetaninDSentNumTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SentenceCountTest, SmetaninDSentNumTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smetanin_d_sent_num
