#include <cuda/barrier>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>

__global__ void async_layer_pipeline(half* g_weights, half* g_kv_cache, int num_layers) {
    extern __shared__ half smem[];
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    
    for (int layer = 0; layer < num_layers; ++layer) {
        pipe.producer_acquire();
        // dma copy khong can cpu dong bo
        cuda::memcpy_async(&smem[0], &g_weights[layer * 256], 256 * sizeof(half), pipe);
        pipe.producer_commit();
        
        pipe.consumer_wait();
        half val = smem[threadIdx.x];
        pipe.consumer_release();
    }
}
