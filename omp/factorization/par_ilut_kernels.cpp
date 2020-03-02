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

#include <algorithm>
#include <unordered_map>
#include <unordered_set>


#include <omp.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *m,
                      IndexType rank, Array<ValueType> &tmp,
                      Array<remove_complex<ValueType>> &,
                      remove_complex<ValueType> &threshold)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    tmp.resize_and_reset(size);
    std::copy_n(values, size, tmp.get_data());

    auto begin = tmp.get_data();
    auto target = begin + rank;
    auto end = begin + size;
    std::nth_element(begin, target, end,
                     [](ValueType a, ValueType b) { return abs(a) < abs(b); });
    threshold = abs(*target);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


template <typename Predicate, typename ValueType, typename IndexType>
void abstract_filter(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *a,
                     matrix::Csr<ValueType, IndexType> *m_out,
                     matrix::Coo<ValueType, IndexType> *m_out_coo,
                     Predicate pred)
{
    auto num_rows = a->get_size()[0];
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

    // first sweep: count nnz for each row
    auto new_row_ptrs = m_out->get_row_ptrs();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType count{};
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; ++nz) {
            count += pred(row, nz);
        }
        new_row_ptrs[row] = count;
    }

    // build row pointers
    prefix_sum(exec, new_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = new_row_ptrs[num_rows];
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
    coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
    coo_builder.get_col_idx_array() =
        Array<IndexType>::view(exec, new_nnz, new_col_idxs);
    coo_builder.get_value_array() =
        Array<ValueType>::view(exec, new_nnz, new_vals);
    auto new_row_idxs = m_out_coo->get_row_idxs();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto new_nz = new_row_ptrs[row];
        auto begin = row_ptrs[row];
        auto end = row_ptrs[row + 1];
        for (auto nz = begin; nz < end; ++nz) {
            if (pred(row, nz)) {
                new_row_idxs[new_nz] = row;
                new_col_idxs[new_nz] = col_idxs[nz];
                new_vals[new_nz] = vals[nz];
                ++new_nz;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType> *m_out,
                      matrix::Coo<ValueType, IndexType> *m_out_coo)
{
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    abstract_filter(
        exec, a, m_out, m_out_coo, [&](IndexType row, IndexType nz) {
            return abs(vals[nz]) >= threshold || col_idxs[nz] == row;
        });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


constexpr auto bucket_count = 256;
constexpr auto oversampling_factor = 4;
constexpr auto sample_size = bucket_count * oversampling_factor;


template <typename ValueType, typename IndexType>
void threshold_filter_approx(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType> *m,
                             IndexType rank, Array<ValueType> &tmp,
                             matrix::Csr<ValueType, IndexType> *m_out,
                             matrix::Coo<ValueType, IndexType> *m_out_coo)
{
    auto vals = m->get_const_values();
    auto col_idxs = m->get_const_col_idxs();
    auto size = static_cast<IndexType>(m->get_num_stored_elements());
    using AbsType = remove_complex<ValueType>;
    auto num_threads = omp_get_max_threads();
    auto storage_size =
        ceildiv(sample_size * sizeof(AbsType) +
                    bucket_count * (num_threads + 1) * sizeof(IndexType),
                sizeof(ValueType));
    tmp.resize_and_reset(storage_size);
    // pick and sort sample
    auto sample = reinterpret_cast<AbsType *>(tmp.get_data());
    // assuming rounding towards zero
    auto stride = double(size) / sample_size;
    for (IndexType i = 0; i < sample_size; ++i) {
        sample[i] = abs(vals[static_cast<IndexType>(i * stride)]);
    }
    std::sort(sample, sample + sample_size);
    // pick splitters
    for (IndexType i = 0; i < bucket_count - 1; ++i) {
        // shift by one so we get upper bounds for the buckets
        sample[i] = sample[(i + 1) * oversampling_factor];
    }
    // count elements per bucket
    auto total_histogram = reinterpret_cast<IndexType *>(sample + bucket_count);
    for (IndexType bucket = 0; bucket < bucket_count; ++bucket) {
        total_histogram[bucket] = 0;
    }
#pragma omp parallel
    {
        auto local_histogram =
            total_histogram + (omp_get_thread_num() + 1) * bucket_count;
        for (IndexType bucket = 0; bucket < bucket_count; ++bucket) {
            local_histogram[bucket] = 0;
        }
#pragma omp for
        for (IndexType nz = 0; nz < size; ++nz) {
            auto bucket_it = std::upper_bound(sample, sample + bucket_count - 1,
                                              abs(vals[nz]));
            auto bucket = std::distance(sample, bucket_it);
            // smallest bucket s.t. sample[bucket] >= abs(val[nz])
            local_histogram[bucket]++;
        }
        for (IndexType bucket = 0; bucket < bucket_count; ++bucket) {
#pragma omp atomic
            total_histogram[bucket] += local_histogram[bucket];
        }
    }
    // determine splitter ranks: prefix sum over bucket counts
    prefix_sum(exec, total_histogram, bucket_count + 1);
    // determine the bucket containing the threshold rank:
    // prefix_sum[bucket] <= rank < prefix_sum[bucket + 1]
    auto it = std::upper_bound(total_histogram,
                               total_histogram + bucket_count + 1, rank);
    auto threshold_bucket = std::distance(total_histogram + 1, it);
    // filter elements
    abstract_filter(
        exec, m, m_out, m_out_coo, [&](IndexType row, IndexType nz) {
            auto bucket_it = std::upper_bound(sample, sample + bucket_count - 1,
                                              abs(vals[nz]));
            auto bucket = std::distance(sample, bucket_it);
            return bucket >= threshold_bucket || col_idxs[nz] == row;
        });
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *a,
                         matrix::Csr<ValueType, IndexType> *l,
                         const matrix::Coo<ValueType, IndexType> *,
                         matrix::Csr<ValueType, IndexType> *u_csc,
                         const matrix::Coo<ValueType, IndexType> *)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_values();
    auto u_col_ptrs = u_csc->get_const_row_ptrs();
    auto u_row_idxs = u_csc->get_const_col_idxs();
    auto u_vals = u_csc->get_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();

    auto compute_sum = [&](IndexType row, IndexType col) {
        // find value from A
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto a_nz_it =
            std::lower_bound(a_col_idxs + a_begin, a_col_idxs + a_end, col);
        auto a_nz = std::distance(a_col_idxs, a_nz_it);
        auto a_val = a_col_idxs[a_nz] == col ? a_vals[a_nz] : zero<ValueType>();
        // accumulate l(row,:) * u(:,col) without the last entry (row, col)
        ValueType sum{};
        auto l_begin = l_row_ptrs[row];
        auto l_end = l_row_ptrs[row + 1];
        auto u_begin = u_col_ptrs[col];
        auto u_end = u_col_ptrs[col + 1];
        auto last_entry = min(row, col);
        while (l_begin < l_end && u_begin < u_end) {
            auto l_col = l_col_idxs[l_begin];
            auto u_row = u_row_idxs[u_begin];
            if (l_col == u_row && l_col < last_entry) {
                sum += l_vals[l_begin] * u_vals[u_begin];
            }
            l_begin += (l_col <= u_row);
            u_begin += (u_row <= l_col);
        }
        return a_val - sum;
    };

#pragma omp parallel for
    for (size_type i = 0; i < num_rows; ++i) {
        for (size_type l_nz = l_row_ptrs[i]; l_nz < l_row_ptrs[i + 1] - 1;
             ++l_nz) {
            auto col = l_col_idxs[l_nz];
            auto u_diag = u_vals[u_col_ptrs[col + 1] - 1];
            auto new_val = compute_sum(i, col) / u_diag;
            if (::gko::isfinite(new_val)) {
                l_vals[l_nz] = new_val;
            }
        }
        for (size_type u_nz = u_col_ptrs[i]; u_nz < u_col_ptrs[i + 1]; ++u_nz) {
            auto row = u_row_idxs[u_nz];
            auto new_val = compute_sum(row, i);
            if (::gko::isfinite(new_val)) {
                u_vals[u_nz] = new_val;
            }
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS);


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *lu,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Csr<ValueType, IndexType> *u,
                    matrix::Csr<ValueType, IndexType> *l_new,
                    matrix::Csr<ValueType, IndexType> *u_new)
{
    auto num_rows = a->get_size()[0];
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto lu_row_ptrs = lu->get_const_row_ptrs();
    auto lu_col_idxs = lu->get_const_col_idxs();
    auto lu_vals = lu->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto u_row_ptrs = u->get_const_row_ptrs();
    auto u_col_idxs = u->get_const_col_idxs();
    auto u_vals = u->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    auto u_new_row_ptrs = u_new->get_row_ptrs();
    auto sentinel = std::numeric_limits<IndexType>::max();
    // count nnz
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType l_row_nnz{};
        IndexType u_row_nnz{};
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto lu_begin = lu_row_ptrs[row];
        auto lu_end = lu_row_ptrs[row + 1];
        auto total_size = (a_end - a_begin) + (lu_end - lu_begin);
        for (IndexType i = 0; i < total_size; ++i) {
            // load column indices or sentinel
            auto a_col = a_begin < a_end ? a_col_idxs[a_begin] : sentinel;
            auto lu_col = lu_begin < lu_end ? lu_col_idxs[lu_begin] : sentinel;
            auto r_col = min(a_col, lu_col);
            // increment row nnz
            l_row_nnz += r_col <= row;
            u_row_nnz += r_col >= row;
            // advance indices
            a_begin += (a_col <= lu_col);
            lu_begin += (lu_col <= a_col);
            i += (a_col == lu_col);
        }
        l_new_row_ptrs[row] = l_row_nnz;
        u_new_row_ptrs[row] = u_row_nnz;
    }

    prefix_sum(exec, l_new_row_ptrs, num_rows + 1);
    prefix_sum(exec, u_new_row_ptrs, num_rows + 1);

    // resize arrays
    auto l_nnz = l_new_row_ptrs[num_rows];
    auto u_nnz = u_new_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> l_builder{l_new};
    matrix::CsrBuilder<ValueType, IndexType> u_builder{u_new};
    l_builder.get_col_idx_array().resize_and_reset(l_nnz);
    l_builder.get_value_array().resize_and_reset(l_nnz);
    u_builder.get_col_idx_array().resize_and_reset(u_nnz);
    u_builder.get_value_array().resize_and_reset(u_nnz);
    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();
    auto u_new_col_idxs = u_new->get_col_idxs();
    auto u_new_vals = u_new->get_values();

    // accumulate non-zeros
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto lu_begin = lu_row_ptrs[row];
        auto lu_end = lu_row_ptrs[row + 1];
        auto l_begin = l_row_ptrs[row];
        auto l_end = l_row_ptrs[row + 1] - 1;  // skip diagonal
        auto u_begin = u_row_ptrs[row];
        auto u_end = u_row_ptrs[row + 1];
        auto l_nz = l_new_row_ptrs[row];
        auto u_nz = u_new_row_ptrs[row];
        auto finished_l = l_begin == l_end;
        auto total_size = (a_end - a_begin) + (lu_end - lu_begin);
        for (IndexType i = 0; i < total_size; ++i) {
            // load column indices or sentinel
            auto a_col = a_begin < a_end ? a_col_idxs[a_begin] : sentinel;
            auto lu_col = lu_begin < lu_end ? lu_col_idxs[lu_begin] : sentinel;
            auto r_col = min(a_col, lu_col);
            // load corresponding values or zero
            auto a_val = a_col <= lu_col ? a_vals[a_begin] : zero<ValueType>();
            auto lu_val =
                lu_col <= a_col ? lu_vals[lu_begin] : zero<ValueType>();
            auto r_val = a_val - lu_val;
            // load matching entry of L + U
            auto lpu_col =
                finished_l ? (u_begin < u_end ? u_col_idxs[u_begin] : sentinel)
                           : l_col_idxs[l_begin];
            auto lpu_val = finished_l ? (u_begin < u_end ? u_vals[u_begin]
                                                         : zero<ValueType>())
                                      : l_vals[l_begin];
            // load diagonal entry of U for lower diagonal entries
            auto diag =
                r_col < row ? u_vals[u_row_ptrs[r_col]] : one<ValueType>();
            // if there is already an entry present, use that instead.
            auto out_val = lpu_col == r_col ? lpu_val : r_val / diag;
            // store output entries
            if (row >= r_col) {
                l_new_col_idxs[l_nz] = r_col;
                l_new_vals[l_nz] = row == r_col ? one<ValueType>() : out_val;
                l_nz++;
            }
            if (row <= r_col) {
                u_new_col_idxs[u_nz] = r_col;
                u_new_vals[u_nz] = out_val;
                ++u_nz;
            }

            // advance indices
            a_begin += (a_col <= lu_col);
            lu_begin += (lu_col <= a_col);
            i += (a_col == lu_col);
            // advance entry of L + U if we used it
            if (finished_l) {
                u_begin += (lpu_col == r_col);
            } else {
                l_begin += (lpu_col == r_col);
                finished_l = (l_begin == l_end);
            }
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
