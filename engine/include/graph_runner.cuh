#pragma once
#include <cuda_runtime.h>

class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    void capture_graph(void* virtual_base_addr, int num_layers);
    void launch(cudaStream_t stream);

private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool is_captured_;
};
