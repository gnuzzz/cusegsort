project(cusegsort)

set(PROJECT_VERSION 1.0.2)

cmake_minimum_required(VERSION 3.11)

FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -arch=sm_35 --expt-relaxed-constexpr)
set(CMAKE_CXX_STANDARD 11)

INCLUDE(FindCUDA)

cuda_add_executable(cusegsort main.cu)
