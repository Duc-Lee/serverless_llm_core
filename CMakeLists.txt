cmake_minimum_required(VERSION 3.18)
project(HydraServerless VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)

include_directories(vmm-core/include)
include_directories(streaming/include)
include_directories(engine/include)

add_library(shadow_vmm STATIC vmm-core/src/shadow_vmm.cpp)
target_link_libraries(shadow_vmm PRIVATE CUDA::cuda_driver)

add_library(gds_loader STATIC streaming/src/gds_loader.cpp)
target_link_libraries(gds_loader PRIVATE CUDA::cuda_driver CUDA::cufile)

add_library(hydra_engine STATIC engine/src/hydra_engine.cpp engine/src/tiered_slicer.cpp engine/src/compute_kernels.cu)
set_target_properties(hydra_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hydra_engine PRIVATE shadow_vmm gds_loader CUDA::cuda_driver)

add_executable(hydra_daemon engine/src/main.cpp)
target_link_libraries(hydra_daemon PRIVATE hydra_engine shadow_vmm gds_loader CUDA::cuda_driver)

add_executable(bench_swap vmm-core/benchmarks/bench_swap.cpp)
target_link_libraries(bench_swap PRIVATE shadow_vmm CUDA::cuda_driver)

