/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/ginkgo.hpp>
#include <iostream>


std::unique_ptr<gko::matrix::Dense<>> generate_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size, double mu)
{
    auto host_matrix = gko::matrix::Dense<>::create(exec->get_master(),
                                                    gko::dim<2>(size, size));

    for (auto i = 0; i < size; i++) {
        for (auto j = 0; j < size; j++) {
            host_matrix->at(i, j) =
                1. / (double)(j + 1 + mu * (i + 1) + size) + (double)(i == j);
        }
    }

    auto matrix = gko::matrix::Dense<>::create(exec);
    matrix->copy_from(host_matrix.get());
    return matrix;
}

std::int64_t mcg_state = 1;

std::uint32_t mcg_rand()
{
    const uint64_t MULTIPLIER = 14647171131086947261U;
    mcg_state *= MULTIPLIER;
    return mcg_state >> 32;
}

double mcg_rand_double()
{
    return (static_cast<double>(mcg_rand())) / UINT32_MAX - 0.5;
}

std::unique_ptr<gko::matrix::Dense<>> generate_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto host_matrix = gko::matrix::Dense<>::create(exec->get_master(),
                                                    gko::dim<2>(size, size));
    auto host_diag = gko::Array<double>(exec->get_master(), size);
    for (auto i = 0; i < size; i++) {
        host_diag.get_data()[i] = 0;
    }

    for (auto j = 0; j < size; j++) {
        for (auto i = 0; i < size; i++) {
            host_matrix->at(i, j) = mcg_rand_double();
            host_diag.get_data()[i] += std::fabs(host_matrix->at(i, j));
        }
    }

    for (auto i = 0; i < size; i++) {
        host_matrix->at(i) =
            host_diag.get_data()[i] - std::fabs(host_matrix->at(i, i));
    }

    auto matrix = gko::matrix::Dense<>::create(exec);
    matrix->copy_from(host_matrix.get());
    return matrix;
}
