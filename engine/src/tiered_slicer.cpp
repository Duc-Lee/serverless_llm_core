#include "tiered_slicer.hpp"

namespace hydra {

TieredSlicer::TieredSlicer(size_t total_weight_size, int num_layers) {
    size_t layer_size = total_weight_size / num_layers;
    
    for (int i = 0; i < num_layers; ++i) {
        layers_.push_back({
            static_cast<size_t>(i),
            i * layer_size,
            layer_size,
            layer_size * 2
        });
    }
    
    pinned_count_ = (num_layers > 2) ? 2 : 0;
}

} // namespace hydra
