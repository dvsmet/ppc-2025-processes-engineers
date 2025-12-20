#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "smetanin_d_gauss_vert_sch/common/include/common.hpp"
#include "smetanin_d_gauss_vert_sch/mpi/include/ops_mpi.hpp"
#include "smetanin_d_gauss_vert_sch/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace smetanin_d_gauss_vert_sch {

namespace {

bool SolveDenseSystem(std::vector<double> a, int n, std::vector<double> &x) {
  const double eps = 1e-10;
  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double best =
        std::fabs(a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)]);
    for (int rr = i + 1; rr < n; ++rr) {
      double val =
          std::fabs(a[(static_cast<std::size_t>(rr) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)]);
      if (val > best) {
        best = val;
        pivot = rr;
      }
    }
    if (best < eps) {
      return false;
    }
    if (pivot != i) {
      for (int cj = i; cj <= n; ++cj) {
        std::swap(
            a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(cj)],
            a[(static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(cj)]);
      }
    }
    for (int rr = i + 1; rr < n; ++rr) {
      double factor =
          a[(static_cast<std::size_t>(rr) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)] /
          a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)];
      if (std::fabs(factor) < eps) {
        continue;
      }
      a[(static_cast<std::size_t>(rr) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)] = 0.0;
      for (int cj = i + 1; cj <= n; ++cj) {
        a[(static_cast<std::size_t>(rr) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(cj)] -=
            factor * a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(cj)];
      }
    }
  }

  x.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double sum = a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(n)];
    for (int cj = i + 1; cj < n; ++cj) {
      sum -= a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(cj)] *
             x[static_cast<std::size_t>(cj)];
    }
    double diag = a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(i)];
    if (std::fabs(diag) < eps) {
      return false;
    }
    x[static_cast<std::size_t>(i)] = sum / diag;
  }
  return true;
}

GaussBandInput LoadSystemFromFile(const std::string &filename) {
  GaussBandInput input;
  const std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_smetanin_d_gauss_vert_sch, filename);
  std::ifstream file(abs_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + abs_path);
  }
  file >> input.n >> input.bandwidth;
  if (input.n <= 0) {
    throw std::runtime_error("Invalid n in file: " + abs_path);
  }
  const std::size_t total = static_cast<std::size_t>(input.n) * static_cast<std::size_t>(input.n + 1);
  input.augmented_matrix.resize(total);
  for (std::size_t i = 0; i < total; ++i) {
    if (!(file >> input.augmented_matrix[i])) {
      throw std::runtime_error("Malformed matrix data in file: " + abs_path);
    }
  }
  return input;
}

}  // namespace

class SmetaninDGaussVertSchFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int idx = std::get<0>(test_param);
    const std::string &file = std::get<1>(test_param);
    std::string name = std::to_string(idx) + "_" + file;
    for (char &ch : name) {
      if (std::isalnum(static_cast<unsigned char>(ch)) == 0 && ch != '_') {
        ch = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &file = std::get<1>(params);

    input_data_ = LoadSystemFromFile(file);

    if (!SolveDenseSystem(input_data_.augmented_matrix, input_data_.n, expected_solution_)) {
      throw std::runtime_error("Failed to compute expected solution for " + file);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_solution_.size()) {
      return false;
    }
    const double eps = 1e-8;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(output_data[i] - expected_solution_[i]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_solution_;
};

namespace {

TEST_P(SmetaninDGaussVertSchFuncTests, SolveBandSystem) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(0, "test_0.txt"), std::make_tuple(1, "test_1.txt"),
                                            std::make_tuple(2, "test_2.txt")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SmetaninDGaussVertSchMPI, InType>(kTestParam, PPC_SETTINGS_smetanin_d_gauss_vert_sch),
    ppc::util::AddFuncTask<SmetaninDGaussVertSchSEQ, InType>(kTestParam, PPC_SETTINGS_smetanin_d_gauss_vert_sch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SmetaninDGaussVertSchFuncTests::PrintFuncTestName<SmetaninDGaussVertSchFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussBandTests, SmetaninDGaussVertSchFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smetanin_d_gauss_vert_sch
