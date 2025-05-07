cmake_minimum_required(VERSION 3.10)
project(ImageFilters LANGUAGES CXX CUDA)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA specific settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Find required packages
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(MPI REQUIRED)

# Common source files
set(COMMON_SOURCES
    bmplib.cpp
)

# Sequential version
add_executable(filter_seq filter.cpp ${COMMON_SOURCES})

# OpenMP version
add_executable(filter_omp filter_omp.cpp ${COMMON_SOURCES})
target_link_libraries(filter_omp OpenMP::OpenMP_CXX)

# MPI version
add_executable(filter_mpi filter_mpi.cpp ${COMMON_SOURCES})
target_link_libraries(filter_mpi MPI::MPI_CXX)

# CUDA version
add_executable(filter_cuda filter_cuda.cu ${COMMON_SOURCES})
set_target_properties(filter_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# Set output directory for all executables
set_target_properties(filter_seq filter_omp filter_mpi filter_cuda
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add custom target to build all versions
add_custom_target(build_all
    DEPENDS filter_seq filter_omp filter_mpi filter_cuda
)

# Installation rules
install(TARGETS filter_seq filter_omp filter_mpi filter_cuda
    RUNTIME DESTINATION bin
)

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
message(STATUS "MPI version: ${MPI_CXX_VERSION}")
message(STATUS "Build targets: filter_seq, filter_omp, filter_mpi, filter_cuda")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin") 