#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void dummy_fp16_gemm(const half* layer_weights, half* scratch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Dummy compute to simulate heavy Tensor Core usage
        half w = layer_weights[idx];
        half s = scratch[idx];
        scratch[idx] = __hadd(w, s);
    }
}

extern "C" void run_dummy_llama_forward(CUstream stream, CUdeviceptr weight_ptr, half* scratch_ptr, size_t size_bytes) {
    int num_elements = size_bytes / sizeof(half);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    cudaStream_t run_stream = reinterpret_cast<cudaStream_t>(stream);
    
    dummy_fp16_gemm<<<blocks, threads, 0, run_stream>>>(
        reinterpret_cast<const half*>(weight_ptr), 
        scratch_ptr, 
        num_elements
    );
}
