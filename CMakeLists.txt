cmake_minimum_required(VERSION 3.28)
project(underhfs LANGUAGES CXX)

option(UNDERHFS_WITH_CUDA "Build CUDA kernels" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings.cpp cpp/tensor_core.cpp)
target_include_directories(_core PRIVATE cpp)
target_compile_definitions(_core PRIVATE UNDERHFS_VERSION="0.1.0")

if(UNDERHFS_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(_core PRIVATE cuda/kernels.cu)
  target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_definitions(_core PRIVATE UNDERHFS_WITH_CUDA=1)
endif()

install(TARGETS _core LIBRARY DESTINATION underhfs)
