#include "../include/graph_runner.cuh"
#include "../kernels/tensor_table.cuh"

__constant__ LayerPointers c_model_offsets[MAX_LAYERS];
__constant__ void* c_virtual_base_addr;

CudaGraphRunner::CudaGraphRunner() : is_captured_(false) {}

CudaGraphRunner::~CudaGraphRunner() {
    if (is_captured_) {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(graph_);
    }
}

__global__ void dummy_layer_compute(int layer_id) {
    half* qkv_ptr = get_qkv_ptr(layer_id);
}

void CudaGraphRunner::capture_graph(void* virtual_base_addr, int num_layers) {
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < num_layers; ++i) {
        dummy_layer_compute<<<1, 256, 0, capture_stream>>>(i);
    }

    cudaStreamEndCapture(capture_stream, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);
    is_captured_ = true;
    cudaStreamDestroy(capture_stream);
}

void CudaGraphRunner::launch(cudaStream_t stream) {
    if (is_captured_) {
        // goi 1 lan tu driver, khong overhead CPU
        cudaGraphLaunch(graph_exec_, stream);
    }
}
