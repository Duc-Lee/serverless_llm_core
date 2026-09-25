#pragma once
#include <cuda_runtime.h>
#include <vector>

// 2. CUDA Graph Multi-Stream Capture (Tránh Kernel Launch Overhead)
class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    // Khởi tạo và capture toàn bộ luồng thực thi (forward pass)
    void capture_graph(void* virtual_base_addr, int num_layers);

    // Kích hoạt toàn bộ Pipeline mà không tốn CPU overhead
    void launch(cudaStream_t stream);

private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool is_captured_;
};
