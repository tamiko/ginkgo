/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/csr.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;

    Csr() : mtx_size(532, 231), rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(std::shared_ptr<Mtx::strategy_type> strategy,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref, strategy);
        square_mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda, strategy);
        dmtx->copy_from(mtx.get());
        square_dmtx = Mtx::create(cuda, strategy);
        square_dmtx->copy_from(square_mtx.get());
        dresult = Vec::create(cuda);
        dresult->copy_from(expected.get());
        dy = Vec::create(cuda);
        dy->copy_from(y.get());
        dalpha = Vec::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(cuda);
        dbeta->copy_from(beta.get());
    }

    void set_up_apply_complex_data(
        std::shared_ptr<ComplexMtx::strategy_type> strategy)
    {
        complex_mtx = ComplexMtx::create(ref, strategy);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        complex_dmtx = ComplexMtx::create(cuda, strategy);
        complex_dmtx->copy_from(complex_mtx.get());
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> cuda;
    };

    matrix_pair gen_unsorted_mtx()
    {
        constexpr int min_nnz_per_row = 2;  // Must be at least 2
        auto local_mtx_ref =
            gen_mtx<Mtx>(mtx_size[0], mtx_size[1], min_nnz_per_row);
        for (size_t row = 0; row < mtx_size[0]; ++row) {
            const auto row_ptrs = local_mtx_ref->get_const_row_ptrs();
            const auto start_row = row_ptrs[row];
            auto col_idx = local_mtx_ref->get_col_idxs() + start_row;
            auto vals = local_mtx_ref->get_values() + start_row;
            const auto nnz_in_this_row = row_ptrs[row + 1] - row_ptrs[row];
            auto swap_idx_dist =
                std::uniform_int_distribution<>(0, nnz_in_this_row - 1);
            // shuffle `nnz_in_this_row / 2` times
            for (size_t perm = 0; perm < nnz_in_this_row; perm += 2) {
                const auto idx1 = swap_idx_dist(rand_engine);
                const auto idx2 = swap_idx_dist(rand_engine);
                std::swap(col_idx[idx1], col_idx[idx2]);
                std::swap(vals[idx1], vals[idx2]);
            }
        }
        auto local_mtx_cuda = Mtx::create(cuda);
        local_mtx_cuda->copy_from(local_mtx_ref.get());

        return {std::move(local_mtx_ref), std::move(local_mtx_cuda)};
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    const gko::dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Csr, StrategyAfterCopyIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    ASSERT_EQ(mtx->get_strategy()->get_name(),
              dmtx->get_strategy()->get_name());
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomatical)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>(), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>(), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>(), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>(), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>());
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(alpha.get(), trans.get(), beta.get(), square_mtx.get());
    dmtx->apply(dalpha.get(), d_trans.get(), dbeta.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>());
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(trans.get(), square_mtx.get());
    dmtx->apply(d_trans.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(d_trans.get()),
                        static_cast<Mtx *>(trans.get()), 0.0);
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_complex_data(std::make_shared<ComplexMtx::automatical>(cuda));

    auto trans = complex_mtx->conj_transpose();
    auto d_trans = complex_dmtx->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx *>(d_trans.get()),
                        static_cast<ComplexMtx *>(trans.get()), 0.0);
}


TEST_F(Csr, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->convert_to(dense_mtx.get());
    dmtx->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToDenseIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->move_to(dense_mtx.get());
    dmtx->move_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Csr, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(cuda);

    mtx->convert_to(ell_mtx.get());
    dmtx->convert_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mtx.get(), dell_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(cuda);

    mtx->move_to(ell_mtx.get());
    dmtx->move_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mtx.get(), dell_mtx.get(), 1e-14);
}

TEST_F(Csr, ConvertToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(cuda);

    mtx->convert_to(sparsity_mtx.get());
    dmtx->convert_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(sparsity_mtx.get(), d_sparsity_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(cuda);

    mtx->move_to(sparsity_mtx.get());
    dmtx->move_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(sparsity_mtx.get(), d_sparsity_mtx.get(), 1e-14);
}


TEST_F(Csr, CalculateMaxNnzPerRowIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    gko::size_type max_nnz_per_row;
    gko::size_type dmax_nnz_per_row;

    gko::kernels::reference::csr::calculate_max_nnz_per_row(ref, mtx.get(),
                                                            &max_nnz_per_row);
    gko::kernels::cuda::csr::calculate_max_nnz_per_row(cuda, dmtx.get(),
                                                       &dmax_nnz_per_row);

    ASSERT_EQ(max_nnz_per_row, dmax_nnz_per_row);
}


TEST_F(Csr, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    mtx->convert_to(coo_mtx.get());
    dmtx->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    mtx->move_to(coo_mtx.get());
    dmtx->move_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Csr, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    mtx->convert_to(sellp_mtx.get());
    dmtx->convert_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx.get(), dsellp_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    mtx->move_to(sellp_mtx.get());
    dmtx->move_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx.get(), dsellp_mtx.get(), 1e-14);
}


TEST_F(Csr, CalculateTotalColsIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    gko::size_type total_cols;
    gko::size_type dtotal_cols;

    gko::kernels::reference::csr::calculate_total_cols(
        ref, mtx.get(), &total_cols, 2, gko::matrix::default_slice_size);
    gko::kernels::cuda::csr::calculate_total_cols(
        cuda, dmtx.get(), &dtotal_cols, 2, gko::matrix::default_slice_size);

    ASSERT_EQ(total_cols, dtotal_cols);
}


TEST_F(Csr, CalculatesNonzerosPerRow)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    gko::Array<gko::size_type> row_nnz(ref, mtx->get_size()[0]);
    gko::Array<gko::size_type> drow_nnz(cuda, dmtx->get_size()[0]);

    gko::kernels::reference::csr::calculate_nonzeros_per_row(ref, mtx.get(),
                                                             &row_nnz);
    gko::kernels::cuda::csr::calculate_nonzeros_per_row(cuda, dmtx.get(),
                                                        &drow_nnz);

    GKO_ASSERT_ARRAY_EQ(&row_nnz, &drow_nnz);
}


TEST_F(Csr, ConvertToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<>;
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        cuda, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->convert_to(hybrid_mtx.get());
    dmtx->convert_to(dhybrid_mtx.get());

    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 1e-14);
}


TEST_F(Csr, MoveToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<>;
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        cuda, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->move_to(hybrid_mtx.get());
    dmtx->move_to(dhybrid_mtx.get());

    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 1e-14);
}


TEST_F(Csr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>());
    bool is_sorted_cuda{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_cuda = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_cuda);
}


TEST_F(Csr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();
    bool is_sorted_cuda{};
    bool is_sorted_ref{};

    is_sorted_ref = uns_mtx.ref->is_sorted_by_column_index();
    is_sorted_cuda = uns_mtx.cuda->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_cuda);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();

    uns_mtx.ref->sort_by_column_index();
    uns_mtx.cuda->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.cuda, 0);
}


}  // namespace
