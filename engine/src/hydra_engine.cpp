#include "hydra_engine.hpp"
#include "tiered_slicer.hpp"
#include <iostream>
#include <chrono>

extern "C" void launch_mock_layer_compute(CUstream stream, size_t simulated_flops);

namespace hydra {

HydraEngine::HydraEngine(int device_id, size_t max_vram_reserve) {
    vmm_ = std::make_unique<ShadowWorkerVMM>(device_id, max_vram_reserve);
    gds_ = std::make_unique<GDSLoader>();
    
    cuStreamCreate(&compute_stream_, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&dma_stream_, CU_STREAM_NON_BLOCKING);
    cuEventCreate(&dma_done_event_, CU_EVENT_DISABLE_TIMING);
}

HydraEngine::~HydraEngine() {
    cuStreamDestroy(compute_stream_);
    cuStreamDestroy(dma_stream_);
    cuEventDestroy(dma_done_event_);
}

void HydraEngine::register_tenant(uint32_t model_hash, const std::string& model_path, size_t weight_size) {
    size_t chunk_id = vmm_->allocate_physical_chunk(weight_size);
    tenants_[model_hash] = {model_hash, model_path, weight_size, chunk_id};
}

void HydraEngine::trigger_inference(uint32_t model_hash, size_t estimated_tokens) {
    auto it = tenants_.find(model_hash);
    if (it == tenants_.end()) {
        std::cerr << "HydraEngine: Tenant not found.\n";
        return;
    }
    const auto& tenant = it->second;

    auto start = std::chrono::high_resolution_clock::now();

    vmm_->map_chunk_to_virtual(tenant.chunk_id);
    auto swap_done = std::chrono::high_resolution_clock::now();

    int num_layers = 32; 
    TieredSlicer slicer(tenant.weight_size, num_layers);
    const auto& layers = slicer.get_layers();

    for (size_t i = 0; i < layers.size(); ++i) {
        if (i >= slicer.get_pinned_layer_count()) {
            gds_->load_weights_async(tenant.model_path, 
                                     vmm_->get_virtual_base() + layers[i].offset_bytes, 
                                     layers[i].offset_bytes, 
                                     layers[i].size_bytes, 
                                     dma_stream_);
            cuEventRecord(dma_done_event_, dma_stream_);
            cuStreamWaitEvent(compute_stream_, dma_done_event_, 0);
        }

        launch_mock_layer_compute(compute_stream_, layers[i].compute_flops);
    }
    
    cuStreamSynchronize(compute_stream_);
    auto e2e_done = std::chrono::high_resolution_clock::now();

    double swap_us = std::chrono::duration_cast<std::chrono::microseconds>(swap_done - start).count();
    double total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e2e_done - start).count();

    std::cout << "Engine: Inference completed. Hash=" << model_hash 
              << ", SwapLatency=" << swap_us << "us, TTFT=" << total_ms << "ms\n";
}

} // namespace hydra
