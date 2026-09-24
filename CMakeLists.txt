cmake_minimum_required(VERSION 3.18)
project(HydraServerless LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

include_directories(vmm/include)
include_directories(pipeline/include)

add_library(vmm_arena STATIC vmm/src/vmm_arena.cpp)
target_link_libraries(vmm_arena PRIVATE CUDA::cuda_driver CUDA::cudart)

add_library(compute_kernels STATIC engine/dummy_llama_forward.cu)
target_link_libraries(compute_kernels PRIVATE CUDA::cuda_driver CUDA::cudart)

add_library(layer_streamer STATIC pipeline/src/layer_streamer.cpp)
target_link_libraries(layer_streamer PRIVATE vmm_arena compute_kernels CUDA::cuda_driver CUDA::cudart)

add_executable(bench_end_to_end tests/bench_end_to_end.cpp)
target_link_libraries(bench_end_to_end PRIVATE layer_streamer vmm_arena compute_kernels CUDA::cuda_driver CUDA::cudart)

add_executable(bench_vmm tests/bench_vmm.cpp)
target_link_libraries(bench_vmm PRIVATE vmm_arena CUDA::cuda_driver CUDA::cudart)

add_executable(peeker_daemon ingress/src/peeker_daemon.cpp)

add_executable(hydra_worker worker/src/hydra_worker.cpp)
target_link_libraries(hydra_worker PRIVATE vmm_arena CUDA::cuda_driver CUDA::cudart)
