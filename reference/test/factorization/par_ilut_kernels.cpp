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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/par_ilut_kernels.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


class ParIlut : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr = gko::matrix::Csr<std::complex<value_type>, index_type>;

    ParIlut()
        : i{0., 1.},
          ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),

          mtx1(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                     {.1, .1, 0., 0.},
                                     {-1., -2., -1., 0.},
                                     {-2., -3., -1., 1.}},
                                    ref)),
          mtx1_expect_thrm2(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                                  {0., .1, 0., 0.},
                                                  {0., -2., -1., 0.},
                                                  {-2., -3., 0., 1.}},
                                                 ref)),
          mtx1_expect_thrm3(gko::initialize<Csr>({{.1, 0., 0., 0.},
                                                  {0., .1, 0., 0.},
                                                  {0., 0., -1., 0.},
                                                  {0., -3., 0., 1.}},
                                                 ref)),
          mtx1_complex(gko::initialize<ComplexCsr>(
              {{.1 + 0. * i, 0. * i, 0. * i, 0. * i},
               {-1. + .1 * i, .1 - i, 0. * i, 0. * i},
               {-1. + i, -2. + .2 * i, -1. - .3 * i, 0. * i},
               {1. - 2. * i, -3. - .1 * i, -1. + .1 * i, .1 + 2. * i}},
              ref)),
          mtx1_expect_complex_thrm(gko::initialize<ComplexCsr>(
              {{.1 + 0. * i, 0. * i, 0. * i, 0. * i},
               {0. * i, .1 - i, 0. * i, 0. * i},
               {-1. + i, -2. + .2 * i, -1. - .3 * i, 0. * i},
               {1. - 2. * i, -3. - .1 * i, 0. * i, .1 + 2. * i}},
              ref)),
          mtx_system(gko::initialize<Csr>({{1., 6., 4., 7.},
                                           {2., -5., 0., 8.},
                                           {.5, -3., 6., 0.},
                                           {.2, -.5, -9., 0.}},
                                          ref)),
          mtx_l(gko::initialize<Csr>({{1., 0., 0., 0.},
                                      {4., 1., 0., 0.},
                                      {-1., 0., 1., 0.},
                                      {0., -3., -1., 1.}},
                                     ref)),
          mtx_u(gko::initialize<Csr>({{2., 0., 1., 1.},
                                      {0., 3., 0., 2.},
                                      {0., 0., .5, 0.},
                                      {0., 0., 0., 4.}},
                                     ref)),
          mtx_lu(gko::initialize<Csr>({{1., 2., 3., 4.},
                                       {0., 6., 7., 8.},
                                       {9., .1, .2, 0.},
                                       {.3, .4, .5, .6}},
                                      ref)),
          mtx_l_add_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                 {4., 1., 0., 0.},
                                                 {-1., -3.1 / 3., 1., 0.},
                                                 {-.05, -3., -1., 1.}},
                                                ref)),
          mtx_u_add_expect(gko::initialize<Csr>({{2., 4., 1., 1.},
                                                 {0., 3., -7., 2.},
                                                 {0., 0., .5, 0.},
                                                 {0., 0., 0., 4.}},
                                                ref)),
          mtx_l_it_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                {2., 1., 0., 0.},
                                                {.5, 0., 1., 0.},
                                                {0., .1, -2.25, 1.}},
                                               ref)),
          mtx_u_it_expect(gko::initialize<Csr>({{1., 0., 0., 0.},
                                                {0., -5., 0., 0.},
                                                {4., 0., 4., 0.},
                                                {7., -6., 0., .6}},
                                               ref))
    {}

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx> &mtx, index_type rank,
                     value_type expected, value_type tolerance = 0.0)
    {
        using ValueType = typename Mtx::value_type;
        gko::remove_complex<ValueType> result{};

        gko::remove_complex<ValueType> res{};
        gko::remove_complex<ValueType> dres{};
        gko::Array<ValueType> tmp(ref);
        gko::Array<gko::remove_complex<ValueType>> tmp2(ref);
        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx.get(), rank, tmp, tmp2, result);

        ASSERT_NEAR(result, expected, tolerance);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter(const std::unique_ptr<Mtx> &mtx, value_type threshold,
                     const std::unique_ptr<Mtx> &expected)
    {
        auto res_mtx = Mtx::create(exec, mtx->get_size());
        auto res_mtx_coo = Coo::create(exec, mtx->get_size());

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, mtx.get(), threshold, res_mtx.get(), res_mtx_coo.get());

        GKO_ASSERT_MTX_EQ_SPARSITY(expected, res_mtx);
        GKO_ASSERT_MTX_NEAR(expected, res_mtx, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx, res_mtx_coo);
        GKO_ASSERT_MTX_NEAR(res_mtx, res_mtx_coo, 0);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter_approx(const std::unique_ptr<Mtx> &mtx, index_type rank,
                            const std::unique_ptr<Mtx> &expected)
    {
        auto res_mtx = Mtx::create(exec, mtx->get_size());
        auto res_mtx_coo = Coo::create(exec, mtx->get_size());

        auto tmp = gko::Array<typename Mtx::value_type>{exec};
        gko::kernels::reference::par_ilut_factorization::
            threshold_filter_approx(ref, mtx.get(), rank, tmp, res_mtx.get(),
                                    res_mtx_coo.get());

        GKO_ASSERT_MTX_EQ_SPARSITY(expected, res_mtx);
        GKO_ASSERT_MTX_NEAR(expected, res_mtx, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx, res_mtx_coo);
        GKO_ASSERT_MTX_NEAR(res_mtx, res_mtx_coo, 0);
    }

    std::complex<value_type> i;

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<Csr> mtx1_expect_thrm2;
    std::unique_ptr<Csr> mtx1_expect_thrm3;
    std::unique_ptr<ComplexCsr> mtx1_complex;
    std::unique_ptr<ComplexCsr> mtx1_expect_complex_thrm;
    std::unique_ptr<Csr> mtx_system;
    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_u;
    std::unique_ptr<Csr> mtx_lu;
    std::unique_ptr<Csr> mtx_l_add_expect;
    std::unique_ptr<Csr> mtx_u_add_expect;
    std::unique_ptr<Csr> mtx_l_it_expect;
    std::unique_ptr<Csr> mtx_u_it_expect;
};


TEST_F(ParIlut, KernelThresholdSelect) { test_select(mtx1, 7, 2.0); }


TEST_F(ParIlut, KernelThresholdSelectMin) { test_select(mtx1, 0, 0.1); }


TEST_F(ParIlut, KernelThresholdSelectMax) { test_select(mtx1, 9, 3.0); }


TEST_F(ParIlut, KernelComplexThresholdSelect)
{
    test_select(mtx1_complex, 5, sqrt(2), 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMin)
{
    test_select(mtx1_complex, 0, 0.1, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMax)
{
    test_select(mtx1_complex, 9, sqrt(9.01), 1e-14);
}


TEST_F(ParIlut, KernelThresholdFilterNone) { test_filter(mtx1, 0.0, mtx1); }


TEST_F(ParIlut, KernelThresholdFilterSomeAtThreshold)
{
    test_filter(mtx1, 2.0, mtx1_expect_thrm2);
}


TEST_F(ParIlut, KernelThresholdFilterSomeAboveThreshold)
{
    test_filter(mtx1, 3.0, mtx1_expect_thrm3);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNone)
{
    test_filter(mtx1_complex, 0.0, mtx1_complex);
}


TEST_F(ParIlut, KernelComplexThresholdFilterSomeAtThreshold)
{
    test_filter(mtx1_complex, 1.01, mtx1_expect_complex_thrm);
}


TEST_F(ParIlut, KernelThresholdFilterSomeApprox1)
{
    test_filter_approx(mtx1, 7, mtx1_expect_thrm2);
}


TEST_F(ParIlut, KernelThresholdFilterSomeApprox2)
{
    test_filter_approx(mtx1, 8, mtx1_expect_thrm2);
}


TEST_F(ParIlut, KernelThresholdFilterNoneApprox)
{
    test_filter_approx(mtx1, 0, mtx1);
}


TEST_F(ParIlut, KernelComplexThresholdFilterSomeApprox)
{
    test_filter_approx(mtx1_complex, 4, mtx1_expect_complex_thrm);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneApprox)
{
    test_filter_approx(mtx1_complex, 0, mtx1_complex);
}


TEST_F(ParIlut, KernelAddCandidates)
{
    auto res_mtx_l = Csr::create(exec, mtx_system->get_size());
    auto res_mtx_u = Csr::create(exec, mtx_system->get_size());

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        ref, mtx_lu.get(), mtx_system.get(), mtx_l.get(), mtx_u.get(),
        res_mtx_l.get(), res_mtx_u.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, mtx_l_add_expect);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_u, mtx_u_add_expect);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, mtx_l_add_expect, 1e-14);
    GKO_ASSERT_MTX_NEAR(res_mtx_u, mtx_u_add_expect, 1e-14);
}

TEST_F(ParIlut, KernelComputeLU)
{
    auto mtx_l_coo = Coo::create(exec, mtx_system->get_size());
    auto mtx_u_transp = mtx_u->transpose();
    auto mtx_u_coo = Coo::create(exec, mtx_system->get_size());
    mtx_l->convert_to(mtx_l_coo.get());
    auto mtx_u_csc = gko::as<Csr>(mtx_u_transp.get());
    mtx_u_csc->convert_to(mtx_u_coo.get());

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        ref, mtx_system.get(), mtx_l.get(), mtx_l_coo.get(), mtx_u_csc,
        mtx_u_coo.get());

    GKO_ASSERT_MTX_NEAR(mtx_l, mtx_l_it_expect, 1e-14);
    GKO_ASSERT_MTX_NEAR(mtx_u_csc, mtx_u_it_expect, 1e-14);
}


}  // namespace
