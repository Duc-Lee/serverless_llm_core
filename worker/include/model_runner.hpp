#pragma once
#include "../../vmm/include/cuda_vmm.hpp"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstddef>

class ModelRunner {
public:
    ModelRunner(VirtualVramManager& vmm);
    ~ModelRunner();

    void forward_layer_gemm(size_t weight_offset, half* d_input, half* d_output, int m, int n, int k, cudaStream_t stream);

private:
    VirtualVramManager& vmm_;
    cublasLtHandle_t cublas_handle_;
};
