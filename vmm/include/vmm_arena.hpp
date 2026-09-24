#pragma once
#include <cuda.h>
#include <vector>
#include <cstddef>

namespace hydra {

class VMMVirtualArena {
public:
    VMMVirtualArena(int device_id, size_t max_virtual_size = 32ULL * 1024 * 1024 * 1024);
    ~VMMVirtualArena();

    CUmemGenericAllocationHandle allocate_physical_chunk(size_t size);
    
    void swap_layer_chunk(size_t virtual_offset, CUmemGenericAllocationHandle handle, size_t size);

    CUdeviceptr get_virtual_base() const { return virtual_base_ptr_; }
    size_t get_granularity() const { return granularity_; }

private:
    CUdevice device_;
    CUcontext context_;
    CUdeviceptr virtual_base_ptr_;
    size_t reservation_size_;
    size_t granularity_;
    std::vector<CUmemGenericAllocationHandle> physical_handles_;
};

} // namespace hydra
