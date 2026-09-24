#pragma once
#include <vector>
#include <cstddef>

namespace hydra {

struct LayerConfig {
    size_t layer_id;
    size_t offset_bytes;
    size_t size_bytes;
    size_t compute_flops;
};

class TieredSlicer {
public:
    TieredSlicer(size_t total_weight_size, int num_layers);
    
    const std::vector<LayerConfig>& get_layers() const { return layers_; }
    size_t get_pinned_layer_count() const { return pinned_count_; }

private:
    std::vector<LayerConfig> layers_;
    size_t pinned_count_;
};

} // namespace hydra
