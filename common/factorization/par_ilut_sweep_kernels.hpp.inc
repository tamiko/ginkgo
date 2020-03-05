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

namespace kernel {


template <typename ValueType>
__forceinline__ __device__ bool is_finite(ValueType v)
{
    return ::gko::isfinite(v);
}


template <int subwarp_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void sweep_sequential(
    const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const ValueType *a_vals, const IndexType *l_row_ptrs,
    const IndexType *l_row_idxs, const IndexType *l_col_idxs, ValueType *l_vals,
    IndexType l_nnz, const IndexType *u_col_ptrs, const IndexType *u_row_idxs,
    const IndexType *u_col_idxs, ValueType *u_vals, IndexType u_nnz)
{
    auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= l_nnz + u_nnz) {
        return;
    }
    auto l_nz = static_cast<IndexType>(tidx);
    auto u_nz = l_nz - l_nnz;
    auto lower = u_nz < 0;
    auto row = lower ? l_row_idxs[l_nz] : u_row_idxs[u_nz];
    auto col = lower ? l_col_idxs[l_nz] : u_col_idxs[u_nz];
    if (lower && row == col) {
        // don't update the diagonal twice
        return;
    }
    auto a_row_begin = a_row_ptrs[row];
    auto a_row_size = a_row_ptrs[row + 1] - a_row_begin;
    auto a_idx = binary_search(a_row_begin, a_row_size, [&](IndexType i) {
        return a_col_idxs[i] >= col;
    });
    auto a_val = a_col_idxs[a_idx] == col ? a_vals[a_idx] : zero<ValueType>();
    auto l_row_begin = l_row_ptrs[row];
    auto l_row_size = l_row_ptrs[row + 1] - l_row_begin;
    auto u_col_begin = u_col_ptrs[col];
    auto u_col_size = u_col_ptrs[col + 1] - u_col_begin;
    ValueType sum{};
    auto last_entry = min(row, col);
    sequential_merge(l_col_idxs + l_row_begin, l_row_size,
                     u_row_idxs + u_col_begin, u_col_size,
                     [&](IndexType l_idx, IndexType l_col, IndexType u_idx,
                         IndexType u_row, IndexType) {
                         if (l_col == u_row && l_col < last_entry) {
                             sum += l_vals[l_idx + l_row_begin] *
                                    u_vals[u_idx + u_col_begin];
                         }
                     });
    if (lower) {
        auto to_write = (a_val - sum) / u_vals[u_col_ptrs[col + 1] - 1];
        if (is_finite(to_write)) {
            l_vals[l_nz] = to_write;
        }
    } else {
        auto to_write = a_val - sum;
        if (is_finite(to_write)) {
            u_vals[u_nz] = to_write;
        }
    }
}


template <int subwarp_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void sweep(
    const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const ValueType *a_vals, const IndexType *l_row_ptrs,
    const IndexType *l_row_idxs, const IndexType *l_col_idxs, ValueType *l_vals,
    IndexType l_nnz, const IndexType *u_col_ptrs, const IndexType *u_row_idxs,
    const IndexType *u_col_idxs, ValueType *u_vals, IndexType u_nnz)
{
    auto tidx = (threadIdx.x + blockIdx.x * blockDim.x) / subwarp_size;
    if (tidx >= l_nnz + u_nnz) {
        return;
    }
    auto l_nz = static_cast<IndexType>(tidx);
    auto u_nz = l_nz - l_nnz;
    auto lower = u_nz < 0;
    auto row = lower ? l_row_idxs[l_nz] : u_row_idxs[u_nz];
    auto col = lower ? l_col_idxs[l_nz] : u_col_idxs[u_nz];
    if (lower && row == col) {
        // don't update the diagonal twice
        return;
    }
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto a_row_begin = a_row_ptrs[row];
    auto a_row_size = a_row_ptrs[row + 1] - a_row_begin;
    auto a_idx =
        group_wide_search(a_row_begin, a_row_size, subwarp,
                          [&](IndexType i) { return a_col_idxs[i] >= col; });
    auto a_val = a_col_idxs[a_idx] == col ? a_vals[a_idx] : zero<ValueType>();
    auto l_row_begin = l_row_ptrs[row];
    auto l_row_size = l_row_ptrs[row + 1] - l_row_begin;
    auto u_col_begin = u_col_ptrs[col];
    auto u_col_size = u_col_ptrs[col + 1] - u_col_begin;
    ValueType sum{};
    auto last_entry = min(row, col);
    group_merge<subwarp_size>(
        l_col_idxs + l_row_begin, l_row_size, u_row_idxs + u_col_begin,
        u_col_size, subwarp,
        [&](IndexType l_idx, IndexType l_col, IndexType u_idx, IndexType u_row,
            IndexType, bool) {
            if (l_col == u_row && l_col < last_entry) {
                sum +=
                    l_vals[l_idx + l_row_begin] * u_vals[u_idx + u_col_begin];
            }
        });
    // accumulate result from all threads
    sum = reduce(subwarp, sum, [](ValueType a, ValueType b) { return a + b; });

    if (subwarp.thread_rank() == 0) {
        if (lower) {
            auto to_write = (a_val - sum) / u_vals[u_col_ptrs[col + 1] - 1];
            if (is_finite(to_write)) {
                l_vals[l_nz] = to_write;
            }
        } else {
            auto to_write = a_val - sum;
            if (is_finite(to_write)) {
                u_vals[u_nz] = to_write;
            }
        }
    }
}


}  // namespace kernel