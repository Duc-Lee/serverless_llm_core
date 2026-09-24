#pragma once
#include "vmm_arena.hpp"
#include <cuda.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace hydra {

struct LayerMeta {
    size_t id;
    size_t size_bytes;
    size_t file_offset;
    CUmemGenericAllocationHandle handle;
};

class LayerStreamer {
public:
    LayerStreamer(VMMVirtualArena& arena, const std::string& weight_file, size_t layer_size, int num_layers);
    ~LayerStreamer();

    void execute_pipeline();

private:
    VMMVirtualArena& arena_;
    std::string weight_file_;
    int num_layers_;
    size_t layer_size_;
    
    std::vector<LayerMeta> layers_;
    
    CUstream compute_stream_;
    CUstream dma_stream_;
    CUevent dma_done_event_;
    
    half* scratch_ptr_;
};

} // namespace hydra
