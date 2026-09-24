#include "shadow_vmm.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    try {
        int device_id = 0;
        size_t virtual_arena_size = 16ULL * 1024 * 1024 * 1024; 
        hydra::ShadowWorkerVMM worker(device_id, virtual_arena_size);
        
        size_t chunk_size = 2ULL * 1024 * 1024 * 1024; 
        size_t chunk0_id = worker.allocate_physical_chunk(chunk_size);
        size_t chunk1_id = worker.allocate_physical_chunk(chunk_size);

        worker.map_chunk_to_virtual(chunk0_id);
        cuCtxSynchronize();

        const int iterations = 10000;
        std::vector<double> latencies_us;
        latencies_us.reserve(iterations);
        
        for (int i = 0; i < iterations; ++i) {
            size_t target_chunk = (i % 2 == 0) ? chunk1_id : chunk0_id;
            
            auto start = std::chrono::high_resolution_clock::now();
            worker.map_chunk_to_virtual(target_chunk);
            auto end = std::chrono::high_resolution_clock::now();
            
            latencies_us.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
        }
        
        std::sort(latencies_us.begin(), latencies_us.end());
        double p50 = latencies_us[iterations / 2];
        double p99 = latencies_us[iterations * 0.99];
        
        double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double avg = sum / iterations;

        std::cout << "Avg Latency: " << avg << " us\n";
        std::cout << "P50 Latency: " << p50 << " us\n";
        std::cout << "P99 Latency: " << p99 << " us\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
