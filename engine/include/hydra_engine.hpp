#pragma once
#include "shadow_vmm.hpp"
#include "gds_loader.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <cuda.h>

namespace hydra {

struct TenantMetadata {
    uint32_t model_hash;
    std::string model_path;
    size_t weight_size;
    size_t chunk_id; 
};

class HydraEngine {
public:
    HydraEngine(int device_id, size_t max_vram_reserve);
    ~HydraEngine();
    
    void register_tenant(uint32_t model_hash, const std::string& model_path, size_t weight_size);
    void trigger_inference(uint32_t model_hash, size_t estimated_tokens);

private:
    std::unique_ptr<ShadowWorkerVMM> vmm_;
    std::unique_ptr<GDSLoader> gds_;
    
    std::unordered_map<uint32_t, TenantMetadata> tenants_;
    
    CUstream compute_stream_;
    CUstream dma_stream_;
    CUevent dma_done_event_;
};

} // namespace hydra
