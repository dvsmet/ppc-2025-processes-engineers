#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "../../common/include/common.hpp"
#include "../../data/matrix_generators.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs {

class SparseMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  InType GetTestInputData() final {
    const auto &param = std::get<2>(GetParam());
    int n = std::get<0>(param);
    double density = std::get<1>(param);

    auto a_matrix = GenerateRandomCRS(n, n, density);
    auto b_matrix = GenerateRandomCRS(n, n, density);

    return {a_matrix, b_matrix};
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
  }
};

TEST_P(SparseMultFuncTests, Run) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 40> kTestParams = {
    {{100, 0.01}, {100, 0.02}, {100, 0.05}, {100, 0.10}, {150, 0.01}, {150, 0.02}, {150, 0.05}, {150, 0.10},
     {200, 0.01}, {200, 0.02}, {200, 0.05}, {200, 0.10}, {250, 0.01}, {250, 0.02}, {250, 0.05}, {250, 0.10},
     {300, 0.01}, {300, 0.02}, {300, 0.05}, {300, 0.10}, {350, 0.01}, {350, 0.02}, {350, 0.05}, {400, 0.01},
     {400, 0.02}, {400, 0.05}, {120, 0.03}, {180, 0.03}, {220, 0.03}, {280, 0.03}, {320, 0.03}, {360, 0.03},
     {380, 0.03}, {100, 0.15}, {150, 0.15}, {200, 0.15}, {250, 0.15}, {300, 0.03}, {350, 0.03}, {400, 0.03}}};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<MultiplicationSparseMatricesCRSMPI, InType>(
                       kTestParams, PPC_SETTINGS_smetanin_d_multiplication_of_sparse_matrices_storage_format_crs),
                   ppc::util::AddFuncTask<MultiplicationSparseMatricesCRSSEQ, InType>(
                       kTestParams, PPC_SETTINGS_smetanin_d_multiplication_of_sparse_matrices_storage_format_crs));

INSTANTIATE_TEST_SUITE_P(FunctionalTests, SparseMultFuncTests, ppc::util::ExpandToValues(kTasks));

}  // namespace smetanin_d_multiplication_of_sparse_matrices_storage_format_crs
