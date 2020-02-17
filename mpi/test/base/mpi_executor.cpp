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

#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include <mpi.h>


namespace {


class MpiExecutor : public ::testing::Test {
protected:
    MpiExecutor() : mpi(nullptr), mpi2(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        // mpi = gko::MpiExecutor::create();
        mpi2 = gko::MpiExecutor::create(argc, argv, 1, true);
    }

    void TearDown()
    {
        if (mpi != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi;
    std::shared_ptr<gko::MpiExecutor> mpi2;
};


TEST_F(MpiExecutor, MpiNonZeroRanks)
{
    ASSERT_GE(mpi2->get_num_ranks(), 1);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ASSERT_EQ(mpi2->get_my_rank(), 1);
}


// TEST_F(MpiExecutor, KnowsCorrectRank)
// {
// }


}  // namespace
