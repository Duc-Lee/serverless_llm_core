#pragma once
#include <vector>
#include <cuda.h>
#include <stdexcept>
#include <string>

#define CUDA_DRIVER_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            throw std::runtime_error("CUDA Driver API error"); \
        } \
    } while (0)

namespace hydra {

class ShadowWorkerVMM {
public:
    struct PhysicalChunk {
        CUmemGenericAllocationHandle handle;
        size_t size;
    };

    ShadowWorkerVMM(int device_id, size_t max_virtual_size);
    ~ShadowWorkerVMM();

    size_t allocate_physical_chunk(size_t size);
    void map_chunk_to_virtual(size_t chunk_id);
    void unmap_virtual();

    CUdeviceptr get_virtual_base() const { return virtual_base_ptr_; }
    size_t get_granularity() const { return granularity_; }

private:
    CUdevice device_;
    CUcontext context_;
    CUdeviceptr virtual_base_ptr_;
    size_t reservation_size_;
    size_t granularity_;
    std::vector<PhysicalChunk> physical_chunks_;
};

} // namespace hydra
