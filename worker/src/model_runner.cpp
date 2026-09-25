#include "../include/model_runner.hpp"

ModelRunner::ModelRunner(VirtualVramManager& vmm) : vmm_(vmm) {
    cublasLtCreate(&cublas_handle_);
}

ModelRunner::~ModelRunner() {
    cublasLtDestroy(cublas_handle_);
}

void ModelRunner::forward_layer_gemm(size_t weight_offset, half* d_input, half* d_output, int m, int n, int k, cudaStream_t stream) {
    // truyen thang base va cong offset thay vi cudaMalloc
    half* d_weight = reinterpret_cast<half*>(vmm_.get_base_va() + weight_offset);
    half alpha = 1.0f;
    half beta  = 0.0f;

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    cublasLtMatrixLayout_t adesc, bdesc, cdesc;
    cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16F, k, n, k);
    cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16F, n, m, n);

    cublasLtMatmul(cublas_handle_, operationDesc, &alpha, d_weight, bdesc, d_input, adesc, &beta, d_output, cdesc, d_output, cdesc, NULL, NULL, 0, stream);

    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
