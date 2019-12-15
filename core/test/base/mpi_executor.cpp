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

#include <ginkgo/core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>


namespace {


// using mpi = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::MpiExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 3;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = 4;
    }

    int &value;
};

class MPITest : public ::testing::Test {
protected:
    using exec = gko::MpiExecutor;
    void SetUp() { mpi = gko::MpiExecutor::create(); }

    // void TearDown() { gko::MpiExecutor::destroy(); }
    void TearDown() {}

    std::shared_ptr<gko::Executor> mpi;
    // std::shared_ptr<gko::MpiExecutor> mpi;
};

// TEST(MpiExecutor, IsItsOwnMaster)
// {
//     exec_ptr mpi = gko::MpiExecutor::create();

//     ASSERT_EQ(mpi, mpi->get_master());
// }


// TEST_F(MPITest, RunsCorrectOperation)
// {
//     int value = 0;
//     // exec_ptr mpi = gko::MpiExecutor::create();

//     mpi->run(ExampleOperation(value));
//     ASSERT_EQ(2, value);
// }


// TEST_F(MPITest, RunsCorrectLambdaOperation)
// {
//     int value = 0;
//     auto omp_lambda = [&value]() { value = 1; };
//     auto mpi_lambda = [&value]() { value = 2; };
//     auto cuda_lambda = [&value]() { value = 4; };
//     // exec_ptr mpi = gko::MpiExecutor::create();

//     mpi->run(omp_lambda, mpi_lambda, cuda_lambda);
//     ASSERT_EQ(2, value);
// }


TEST_F(MPITest, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    // exec_ptr mpi = gko::MpiExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = mpi->alloc<int>(num_elems));
    ASSERT_NO_THROW(mpi->free(ptr));
}


TEST_F(MPITest, FreeAcceptsNullptr)
{
    // exec_ptr mpi = gko::MpiExecutor::create();
    ASSERT_NO_THROW(mpi->free(nullptr));
}


TEST_F(MPITest, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    // exec_ptr mpi = gko::MpiExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = mpi->alloc<int>(num_elems), gko::AllocationError);

    mpi->free(ptr);
}


TEST_F(MPITest, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    // exec_ptr mpi = gko::MpiExecutor::create();
    int *copy = mpi->alloc<int>(num_elems);

    // user code is run on the MPI, so local variables are in MPI memory
    mpi->copy_from(mpi.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    mpi->free(copy);
}


}  // namespace
