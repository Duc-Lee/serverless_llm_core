cmake_minimum_required(VERSION 3.18)
project(HydraServerless LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} vmm/include)

# Tùy chọn cờ biên dịch thư viện Driver API
set(CUDA_DRIVER_LIBRARY /usr/local/cuda/lib64/stubs/libcuda.so)

add_executable(hydra_worker 
    worker/src/main.cpp 
    worker/src/model_runner.cpp
    vmm/src/cuda_vmm.cpp
)
target_link_libraries(hydra_worker ${CUDA_DRIVER_LIBRARY} pthread cublas cublasLt)

# Benchmark riêng cho VMM
add_executable(bench_vmm tests/bench_vmm.cpp vmm/src/cuda_vmm.cpp)
target_link_libraries(bench_vmm ${CUDA_DRIVER_LIBRARY})
