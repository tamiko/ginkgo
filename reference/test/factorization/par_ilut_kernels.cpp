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
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class ParIlut : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr =
        gko::matrix::Csr<std::complex<gko::remove_complex<value_type>>,
                         index_type>;

    ParIlut()
        : ref(gko::ReferenceExecutor::create()),
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
              {{{.1, 0.}, {0., 0.}, {0., 0.}, {0., 0.}},
               {{-1., .1}, {.1, -1.}, {0., 0.}, {0., 0.}},
               {{-1., 1.}, {-2., .2}, {-1., -.3}, {0., 0.}},
               {{1., -2.}, {-3., -.1}, {-1., .1}, {.1, 2.}}},
              ref)),
          mtx1_expect_complex_thrm(gko::initialize<ComplexCsr>(
              {{{.1, 0.}, {0., 0.}, {0., 0.}, {0., 0.}},
               {{0., 0.}, {.1, -1.}, {0., 0.}, {0., 0.}},
               {{-1., 1.}, {-2., .2}, {-1., -.3}, {0., 0.}},
               {{1., -2.}, {-3., -.1}, {0., 0.}, {.1, 2.}}},
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
                     gko::remove_complex<value_type> expected,
                     gko::remove_complex<value_type> tolerance = 0.0)
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
    void test_filter(const std::unique_ptr<Mtx> &mtx,
                     gko::remove_complex<value_type> threshold,
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
};  // namespace

TYPED_TEST_CASE(ParIlut, gko::test::ValueIndexTypes);


TYPED_TEST(ParIlut, KernelThresholdSelect)
{
    this->test_select(this->mtx1, 7, 2.0);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMin)
{
    this->test_select(this->mtx1, 0, 0.1);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMax)
{
    this->test_select(this->mtx1, 9, 3.0);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelect)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 5, sqrt(2), r<value_type>::value);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelectMin)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 0, 0.1, r<value_type>::value);
}


TYPED_TEST(ParIlut, KernelComplexThresholdSelectMax)
{
    using value_type = typename TestFixture::value_type;
    this->test_select(this->mtx1_complex, 9, sqrt(9.01), r<value_type>::value);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNone)
{
    this->test_filter(this->mtx1, 0.0, this->mtx1);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAtThreshold)
{
    this->test_filter(this->mtx1, 2.0, this->mtx1_expect_thrm2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeAboveThreshold)
{
    this->test_filter(this->mtx1, 3.0, this->mtx1_expect_thrm3);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterNone)
{
    this->test_filter(this->mtx1_complex, 0.0, this->mtx1_complex);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterSomeAtThreshold)
{
    this->test_filter(this->mtx1_complex, 1.01, this->mtx1_expect_complex_thrm);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeApprox1)
{
    this->test_filter_approx(this->mtx1, 7, this->mtx1_expect_thrm2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterSomeApprox2)
{
    this->test_filter_approx(this->mtx1, 8, this->mtx1_expect_thrm2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneApprox)
{
    this->test_filter_approx(this->mtx1, 0, this->mtx1);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterSomeApprox)
{
    this->test_filter_approx(this->mtx1_complex, 4,
                             this->mtx1_expect_complex_thrm);
}


TYPED_TEST(ParIlut, KernelComplexThresholdFilterNoneApprox)
{
    this->test_filter_approx(this->mtx1_complex, 0, this->mtx1_complex);
}


TYPED_TEST(ParIlut, KernelAddCandidates)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto res_mtx_l = Csr::create(this->exec, this->mtx_system->get_size());
    auto res_mtx_u = Csr::create(this->exec, this->mtx_system->get_size());

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        this->ref, this->mtx_lu.get(), this->mtx_system.get(),
        this->mtx_l.get(), this->mtx_u.get(), res_mtx_l.get(), res_mtx_u.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, this->mtx_l_add_expect);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_u, this->mtx_u_add_expect);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, this->mtx_l_add_expect,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(res_mtx_u, this->mtx_u_add_expect,
                        r<value_type>::value);
}

TYPED_TEST(ParIlut, KernelComputeLU)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    auto mtx_l_coo = Coo::create(this->exec, this->mtx_system->get_size());
    auto mtx_u_transp = this->mtx_u->transpose();
    auto mtx_u_coo = Coo::create(this->exec, this->mtx_system->get_size());
    this->mtx_l->convert_to(mtx_l_coo.get());
    auto mtx_u_csc = gko::as<Csr>(mtx_u_transp.get());
    mtx_u_csc->convert_to(mtx_u_coo.get());

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        this->ref, this->mtx_system.get(), this->mtx_l.get(), mtx_l_coo.get(),
        mtx_u_csc, mtx_u_coo.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l, this->mtx_l_it_expect,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(mtx_u_csc, this->mtx_u_it_expect, r<value_type>::value);
}


}  // namespace
