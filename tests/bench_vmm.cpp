#include "cuda_vmm.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    try {
        const size_t RESERVED_VA = 32ULL * 1024 * 1024 * 1024;
        const size_t CHUNK_SIZE = 2ULL * 1024 * 1024;
        VirtualVramManager vmm(RESERVED_VA, CHUNK_SIZE);

        std::cout << "Benchmarking map_backing_store(512MB)...\n";
        auto start = std::chrono::high_resolution_clock::now();
        vmm.map_backing_store(512ULL * 1024 * 1024);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Time to map 512MB: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << " us\n";

        vmm.evict_all();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
