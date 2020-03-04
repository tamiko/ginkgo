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

#include "core/factorization/par_ilut_kernels.hpp"


#include <hip/hip_runtime.h>


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
#include "hip/components/searching.hip.hpp"
#include "hip/components/sorting.hip.hpp"
#include "hip/factorization/par_ilut_select_common.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr auto default_block_size = 512;
constexpr auto items_per_thread = 2;


#include "common/factorization/par_ilut_select_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void ssss_count(const ValueType *values, IndexType size,
                remove_complex<ValueType> *tree, unsigned char *oracles,
                IndexType *partial_counts, IndexType *total_counts)
{
    constexpr auto bucket_count = kernel::searchtree_width;
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    // pick sample, build searchtree
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::build_searchtree), dim3(1),
                       dim3(bucket_count), 0, 0, as_hip_type(values), size,
                       tree);
    // determine bucket sizes
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::count_buckets), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, as_hip_type(values),
                       size, tree, partial_counts, oracles, items_per_thread);
    // compute prefix sum and total sum over block-local values
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::block_prefix_sum),
                       dim3(bucket_count), dim3(default_block_size), 0, 0,
                       partial_counts, total_counts, num_blocks);
    // compute prefix sum over bucket counts
    hipLaunchKernelGGL(HIP_KERNEL_NAME(start_prefix_sum<bucket_count>), dim3(1),
                       dim3(bucket_count), 0, 0, bucket_count, total_counts,
                       total_counts + bucket_count);
}


template <typename ValueType, typename IndexType>
void ssss_filter(const ValueType *values, IndexType size,
                 const unsigned char *oracles, const IndexType *partial_counts,
                 IndexType bucket, remove_complex<ValueType> *out)
{
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::filter_bucket), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, as_hip_type(values),
                       size, bucket, oracles, partial_counts, out,
                       items_per_thread);
}


template <typename IndexType>
ssss_bucket<IndexType> ssss_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum,
    IndexType rank)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::find_bucket), dim3(1),
                       dim3(config::warp_size), 0, 0, prefix_sum, rank);
    IndexType values[3]{};
    exec->get_master()->copy_from(exec.get(), 3, prefix_sum, values);
    return {values[0], values[1], values[2]};
}


template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *m,
                      IndexType rank, Array<ValueType> &tmp1,
                      Array<remove_complex<ValueType>> &tmp2,
                      remove_complex<ValueType> &threshold)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    using AbsType = remove_complex<ValueType>;
    constexpr auto bucket_count = kernel::searchtree_width;
    auto max_num_threads = ceildiv(size, items_per_thread);
    auto max_num_blocks = ceildiv(max_num_threads, default_block_size);

    size_type tmp_size_totals =
        ceildiv((bucket_count + 1) * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_partials = ceildiv(
        bucket_count * max_num_blocks * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_oracles =
        ceildiv(size * sizeof(unsigned char), sizeof(ValueType));
    size_type tmp_size_tree =
        ceildiv(kernel::searchtree_size * sizeof(AbsType), sizeof(ValueType));
    size_type tmp_size_vals =
        size / bucket_count * 4;  // pessimistic estimate for temporary storage
    size_type tmp_size =
        tmp_size_totals + tmp_size_partials + tmp_size_oracles + tmp_size_tree;
    tmp1.resize_and_reset(tmp_size);
    tmp2.resize_and_reset(tmp_size_vals);

    auto total_counts = reinterpret_cast<IndexType *>(tmp1.get_data());
    auto partial_counts =
        reinterpret_cast<IndexType *>(tmp1.get_data() + tmp_size_totals);
    auto oracles = reinterpret_cast<unsigned char *>(
        tmp1.get_data() + tmp_size_totals + tmp_size_partials);
    auto tree =
        reinterpret_cast<AbsType *>(tmp1.get_data() + tmp_size_totals +
                                    tmp_size_partials + tmp_size_oracles);

    ssss_count(values, size, tree, oracles, partial_counts, total_counts);

    // determine bucket with correct rank
    auto bucket = ssss_find_bucket(exec, total_counts, rank);
    rank -= bucket.begin;

    if (bucket.size * 2 > tmp_size_vals) {
        // we need to reallocate tmp2
        tmp2.resize_and_reset(bucket.size * 2);
    }
    auto tmp21 = tmp2.get_data();
    auto tmp22 = tmp2.get_data() + bucket.size;
    // extract target bucket
    ssss_filter(values, size, oracles, partial_counts, bucket.idx, tmp22);

    // recursively select from smaller buckets
    int step{};
    while (bucket.size > kernel::basecase_size) {
        std::swap(tmp21, tmp22);
        const auto *tmp_in = tmp21;
        auto tmp_out = tmp22;

        ssss_count(tmp_in, bucket.size, tree, oracles, partial_counts,
                   total_counts);
        auto new_bucket = ssss_find_bucket(exec, total_counts, rank);
        ssss_filter(tmp_in, bucket.size, oracles, partial_counts, bucket.idx,
                    tmp_out);

        rank -= new_bucket.begin;
        bucket.size = new_bucket.size;
        // we should never need more than 5 recursion steps, this would mean
        // 256^5 = 2^40. fall back to standard library algorithm in that case.
        ++step;
        if (step > 5) {
            Array<AbsType> cpu_out_array{
                exec->get_master(),
                Array<AbsType>::view(exec, bucket.size, tmp_out)};
            auto begin = cpu_out_array.get_data();
            auto end = begin + bucket.size;
            auto middle = begin + rank;
            std::nth_element(begin, middle, end);
            threshold = *middle;
            return;
        }
    }

    // base case
    auto out_ptr = reinterpret_cast<AbsType *>(tmp1.get_data());
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::basecase_select), dim3(1),
                       dim3(kernel::basecase_block_size), 0, 0, tmp22,
                       bucket.size, rank, out_ptr);
    exec->get_master()->copy_from(exec.get(), 1, out_ptr, &threshold);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko