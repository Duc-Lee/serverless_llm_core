#include "layer_streamer.hpp"
#include "vmm_arena.hpp"
#include <iostream>

int main() {
    std::cout << "Starting End-to-End Pipelined Benchmark...\n";
    try {
        hydra::VMMVirtualArena arena(0);
        size_t layer_size = 256 * 1024 * 1024; // 256MB per layer
        int num_layers = 16;
        
        hydra::LayerStreamer streamer(arena, "dummy_weights.bin", layer_size, num_layers);
        streamer.execute_pipeline();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
