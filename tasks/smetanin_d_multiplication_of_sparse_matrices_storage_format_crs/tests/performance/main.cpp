#include <gtest/gtest.h>

#include "../../common/include/common.hpp"
#include "../../data/matrix_generators.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

class SparseMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int n = 3000;
    const double density = 0.01;

    auto a_matrix = GenerateRandomCRS(n, n, density);
    auto b_matrix = GenerateRandomCRS(n, n, density);

    input_data_ = {a_matrix, b_matrix};
  }

  bool CheckTestOutputData(OutType & /*unused_output*/) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(SparseMultPerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MultiplicationSparseMatricesCRSMPI, MultiplicationSparseMatricesCRSSEQ>(
        PPC_SETTINGS_smetanin_d_multiplication_of_sparse_matrices_storage_format_crs);

INSTANTIATE_TEST_SUITE_P(PerformanceTests, SparseMultPerfTests, ppc::util::TupleToGTestValues(kPerfTasks));

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
