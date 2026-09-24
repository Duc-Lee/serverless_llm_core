#include "vmm_arena.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

int main() {
    try {
        std::cout << "=========================================\n";
        std::cout << "   VMM Re-map Latency Microbenchmark     \n";
        std::cout << "=========================================\n";
        
        // Reserve 32GB Virtual Address Space
        hydra::VMMVirtualArena arena(0);
        
        std::vector<size_t> chunk_sizes = {
            2ULL * 1024 * 1024,        // 2MB
            64ULL * 1024 * 1024,       // 64MB
            512ULL * 1024 * 1024,      // 512MB
            2ULL * 1024 * 1024 * 1024  // 2GB
        };
        
        for (size_t size : chunk_sizes) {
            CUmemGenericAllocationHandle handle = arena.allocate_physical_chunk(size);
            
            const int iterations = 1000;
            std::vector<double> latencies_us;
            latencies_us.reserve(iterations);
            
            for (int i = 0; i < iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                // Map/Unmap at virtual offset 0
                arena.swap_layer_chunk(0, handle, size);
                auto end = std::chrono::high_resolution_clock::now();
                
                latencies_us.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
            }
            
            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[iterations / 2];
            double p99 = latencies_us[iterations * 0.99];
            
            std::cout << "Chunk Size: " << size / (1024.0 * 1024.0) << " MB | "
                      << "P50 Latency: " << p50 << " us | "
                      << "P99 Latency: " << p99 << " us\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
