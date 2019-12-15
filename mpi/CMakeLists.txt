find_package(MPI REQUIRED)

add_library(ginkgo_mpi $<TARGET_OBJECTS:ginkgo_mpi_device> "")
target_sources(ginkgo_mpi
    PRIVATE
        base/exception.cpp
        base/executor.cpp
        base/version.cpp
        matrix/coo_kernels.cpp
        matrix/csr_kernels.cpp
        matrix/dense_kernels.cpp
        matrix/ell_kernels.cpp
        matrix/hybrid_kernels.cpp
        matrix/sellp_kernels.cpp
        preconditioner/jacobi_kernels.cpp
        solver/bicgstab_kernels.cpp
        solver/cg_kernels.cpp
        solver/cgs_kernels.cpp
        solver/fcg_kernels.cpp
        solver/gmres_kernels.cpp
        solver/ir_kernels.cpp
        stop/criterion_kernels.cpp
        stop/residual_norm_reduction_kernels.cpp)

ginkgo_compile_features(ginkgo_mpi)
target_include_directories(ginkgo_mpi
    SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(ginkgo_mpi PRIVATE "${MPI_C_LIBRARIES}")
target_link_libraries(ginkgo_mpi PRIVATE "${MPI_CXX_LIBRARIES}")
target_compile_options(ginkgo_mpi PRIVATE "${GINKGO_COMPILER_FLAGS}")

# Need to link against ginkgo_cuda for the `raw_copy_to(CudaExecutor ...)` method
target_link_libraries(ginkgo_mpi PUBLIC ginkgo_cuda)

ginkgo_default_includes(ginkgo_mpi)
ginkgo_install_library(ginkgo_mpi mpi)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Propagate some useful information
set(MPI_C_VERSION ${MPI_C_VERSION} PARENT_SCOPE)
set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
