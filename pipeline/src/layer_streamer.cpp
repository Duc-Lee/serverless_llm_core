#include "layer_streamer.hpp"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>

#define CUDA_DRIVER_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            throw std::runtime_error("CUDA Driver API error"); \
        } \
    } while (0)

extern "C" void run_dummy_llama_forward(CUstream stream, CUdeviceptr weight_ptr, half* scratch_ptr, size_t size_bytes);

namespace hydra {

LayerStreamer::LayerStreamer(VMMVirtualArena& arena, const std::string& weight_file, size_t layer_size, int num_layers)
    : arena_(arena), weight_file_(weight_file), layer_size_(layer_size), num_layers_(num_layers) {
    
    CUDA_DRIVER_CHECK(cuStreamCreate(&compute_stream_, CU_STREAM_NON_BLOCKING));
    CUDA_DRIVER_CHECK(cuStreamCreate(&dma_stream_, CU_STREAM_NON_BLOCKING));
    CUDA_DRIVER_CHECK(cuEventCreate(&dma_done_event_, CU_EVENT_DISABLE_TIMING));
    
    cudaMalloc(&scratch_ptr_, layer_size);

    for (int i = 0; i < num_layers; ++i) {
        CUmemGenericAllocationHandle handle = arena_.allocate_physical_chunk(layer_size);
        layers_.push_back({static_cast<size_t>(i), layer_size, i * layer_size, handle});
    }
}

LayerStreamer::~LayerStreamer() {
    cuStreamDestroy(compute_stream_);
    cuStreamDestroy(dma_stream_);
    cuEventDestroy(dma_done_event_);
    cudaFree(scratch_ptr_);
}

void LayerStreamer::execute_pipeline() {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (num_layers_ > 0) arena_.swap_layer_chunk(layers_[0].id * layer_size_, layers_[0].handle, layer_size_);
    if (num_layers_ > 1) arena_.swap_layer_chunk(layers_[1].id * layer_size_, layers_[1].handle, layer_size_);

    for (int i = 0; i < num_layers_; ++i) {
        CUdeviceptr layer_vptr = arena_.get_virtual_base() + layers_[i].id * layer_size_;

        if (i >= 2) {
            arena_.swap_layer_chunk(layers_[i].id * layer_size_, layers_[i].handle, layer_size_);
            // Emulate async GDS/cuFile load
            CUDA_DRIVER_CHECK(cuEventRecord(dma_done_event_, dma_stream_));
            CUDA_DRIVER_CHECK(cuStreamWaitEvent(compute_stream_, dma_done_event_, 0));
        }

        run_dummy_llama_forward(compute_stream_, layer_vptr, scratch_ptr_, layer_size_);
    }

    CUDA_DRIVER_CHECK(cuStreamSynchronize(compute_stream_));
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double ttft_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "[Pipeline] Total Compute/DMA Overlapped TTFT Latency: " << ttft_ms << " ms\n";
}

} // namespace hydra
