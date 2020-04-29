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

#ifndef GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_ISAI_GENERATE_L_INVERSE_KERNEL(ValueType, IndexType)    \
    void generate_l_inverse(std::shared_ptr<const DefaultExecutor> exec,    \
                            const matrix::Csr<ValueType, IndexType> *l_csr, \
                            matrix::Csr<ValueType, IndexType> *inverse_l)

#define GKO_DECLARE_ISAI_GENERATE_U_INVERSE_KERNEL(ValueType, IndexType)    \
    void generate_u_inverse(std::shared_ptr<const DefaultExecutor> exec,    \
                            const matrix::Csr<ValueType, IndexType> *u_csr, \
                            matrix::Csr<ValueType, IndexType> *inverse_u)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_ISAI_GENERATE_L_INVERSE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_ISAI_GENERATE_U_INVERSE_KERNEL(ValueType, IndexType)


namespace omp {
namespace isai {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace isai
}  // namespace omp


namespace cuda {
namespace isai {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace isai
}  // namespace cuda


namespace reference {
namespace isai {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace isai
}  // namespace reference


namespace hip {
namespace isai {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace isai
}  // namespace hip


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_
