#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "smetanin_d_gauss_vert_sch/common/include/common.hpp"
#include "smetanin_d_gauss_vert_sch/mpi/include/ops_mpi.hpp"
#include "smetanin_d_gauss_vert_sch/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smetanin_d_gauss_vert_sch {

namespace {

GaussBandInput MakePerfSystem(int n, int bandwidth) {
  GaussBandInput input;
  input.n = n;
  input.bandwidth = bandwidth;
  input.augmented_matrix.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1), 0.0);

  std::vector<double> x_true(static_cast<std::size_t>(n), 1.0);

  for (int ii = 0; ii < n; ++ii) {
    for (int jj = std::max(0, ii - bandwidth); jj <= std::min(n - 1, ii + bandwidth); ++jj) {
      double value = (ii == jj) ? static_cast<double>((2 * bandwidth) + 1) : 1.0;
      std::size_t idx = (static_cast<std::size_t>(ii) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(jj);
      input.augmented_matrix[idx] = value;
    }
  }

  for (int ii = 0; ii < n; ++ii) {
    double b_val = 0.0;
    for (int jj = std::max(0, ii - bandwidth); jj <= std::min(n - 1, ii + bandwidth); ++jj) {
      std::size_t idx = (static_cast<std::size_t>(ii) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(jj);
      b_val += input.augmented_matrix[idx] * x_true[static_cast<std::size_t>(jj)];
    }
    std::size_t b_idx = (static_cast<std::size_t>(ii) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(n);
    input.augmented_matrix[b_idx] = b_val;
  }

  return input;
}

}  // namespace

class SmetaninDRunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const int n = 2000;
    const int bw = 5;
    input_data_ = MakePerfSystem(n, bw);
    expected_output_.assign(static_cast<std::size_t>(n), 1.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    const double eps = 1e-8;
    for (std::size_t idx = 0; idx < output_data.size(); ++idx) {
      if (std::fabs(output_data[idx] - expected_output_[idx]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmetaninDRunPerfTestProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SmetaninDGaussVertSchMPI, SmetaninDGaussVertSchSEQ>(
    PPC_SETTINGS_smetanin_d_gauss_vert_sch);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmetaninDRunPerfTestProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmetaninDRunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace smetanin_d_gauss_vert_sch
