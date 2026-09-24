#include <cuda_runtime.h>
#include <stdint.h>

__global__ void spin_wait_kernel(uint64_t clock_cycles) {
    uint64_t start = clock64();
    while (clock64() - start < clock_cycles) {
        // Spin wait to simulate compute delay
    }
}

extern "C" void launch_mock_layer_compute(CUstream stream, size_t simulated_flops) {
    uint64_t cycles = simulated_flops / 5000; 
    cudaStream_t run_stream = reinterpret_cast<cudaStream_t>(stream);
    spin_wait_kernel<<<1, 1, 0, run_stream>>>(cycles);
}
