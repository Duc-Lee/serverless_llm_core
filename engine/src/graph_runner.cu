#include "../include/graph_runner.cuh"
#include "../kernels/tensor_table.cuh"
#include <iostream>

// Định nghĩa __constant__ thực tế
__constant__ LayerPointers c_model_offsets[MAX_LAYERS];
__constant__ void* c_virtual_base_addr;

CudaGraphRunner::CudaGraphRunner() : is_captured_(false) {}

CudaGraphRunner::~CudaGraphRunner() {
    if (is_captured_) {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(graph_);
    }
}

// Dummy kernel minh họa gọi trực tiếp từ Constant Memory
__global__ void dummy_layer_compute(int layer_id) {
    // Zero-overhead lookup
    half* qkv_ptr = get_qkv_ptr(layer_id);
    // (Thực thi FlashAttention hoặc GEMM tại đây)
}

void CudaGraphRunner::capture_graph(void* virtual_base_addr, int num_layers) {
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    // Bắt đầu capture toàn bộ pipeline vào 1 Node duy nhất
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < num_layers; ++i) {
        // Ghi nhận các kernel vào Graph thay vì ném thẳng lên SM
        dummy_layer_compute<<<1, 256, 0, capture_stream>>>(i);
    }

    cudaStreamEndCapture(capture_stream, &graph_);
    // Pre-bake execution path
    cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);
    is_captured_ = true;

    cudaStreamDestroy(capture_stream);
}

void CudaGraphRunner::launch(cudaStream_t stream) {
    if (is_captured_) {
        // Bắn 1 phát là chạy hết mảng nơ-ron (Zero CPU Launch Overhead)
        cudaGraphLaunch(graph_exec_, stream);
    }
}
