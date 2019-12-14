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

#include <chrono>
#include <ginkgo/ginkgo.hpp>
#include <iostream>

#include "generate_matrix.hpp"

void run(unsigned int size, double mu, std::shared_ptr<gko::matrix::Dense<>> A,
         std::shared_ptr<const gko::Executor> gpu)
{
    auto host_b =
        gko::matrix::Dense<>::create(gpu->get_master(), gko::dim<2>(size, 1));
    auto host_x =
        gko::matrix::Dense<>::create(gpu->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_b->at(i, 0) = 1.;
        host_x->at(i, 0) = 0.;
    }
    auto x = gko::matrix::Dense<>::create(gpu);
    x->copy_from(host_x.get());
    auto b = gko::matrix::Dense<>::create(gpu);
    b->copy_from(host_b.get());

    auto solver_factory =
        gko::solver::Gmres<>::build()
            .with_criteria(
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-10)
                    .on(gpu),
                gko::stop::Iteration::build().with_max_iters(size).on(gpu))
            .on(gpu);

    auto start_generation = std::chrono::high_resolution_clock::now();
    auto solver = solver_factory->generate(clone(A));
    gpu->synchronize();
    auto end_generation = std::chrono::high_resolution_clock::now();

    auto start_computation = std::chrono::high_resolution_clock::now();
    solver->apply(lend(b), lend(x));
    gpu->synchronize();
    auto end_computation = std::chrono::high_resolution_clock::now();

    std::cout.precision(8);
    auto duration_generation =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_generation -
                                                             start_generation)
            .count() /
        (double)1000000000;

    auto duration_computation =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_computation -
                                                             start_computation)
            .count() /
        (double)1000000000;

    auto neg_one_op =
        gko::initialize<gko::matrix::Dense<>>({-gko::one<double>()}, gpu);
    auto one_op =
        gko::initialize<gko::matrix::Dense<>>({gko::one<double>()}, gpu);
    auto res_norm =
        gko::initialize<gko::matrix::Dense<>>({gko::zero<double>()}, gpu);
    A->apply(lend(one_op), lend(x), lend(neg_one_op), lend(b));
    b->compute_norm2(res_norm.get());

    auto norm = gko::matrix::Dense<>::create(gpu->get_master());

    norm->copy_from(res_norm.get());

    std::cout << "  " << norm->at(0, 0) << "  " << duration_generation << "  "
              << duration_computation << std::endl;
}

int main(int argc, char *argv[])
{
    auto omp = gko::OmpExecutor::create();
    auto gpu = gko::CudaExecutor::create(0, omp);
    unsigned int size = atoi(argv[1]);
    double mu = 0.0;
    std::unique_ptr<gko::matrix::Dense<>> A;
    if (argc == 3) {
        mu = atof(argv[2]);
        A = generate_matrix(gpu, size, mu);
    } else {
        A = generate_matrix(omp, size);
        std::cout << "Not using mu" << std::endl;
    }

    for (auto i = 0; i < 100; i++) {
        run(size, mu, clone(A), omp);
    }
}
