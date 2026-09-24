#include "vmm_arena.hpp"
#include <stdexcept>
#include <string>

#define CUDA_DRIVER_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            throw std::runtime_error("CUDA Driver API error: " + std::to_string(result)); \
        } \
    } while (0)

namespace hydra {

VMMVirtualArena::VMMVirtualArena(int device_id, size_t max_virtual_size) {
    CUDA_DRIVER_CHECK(cuInit(0));
    CUDA_DRIVER_CHECK(cuDeviceGet(&device_, device_id));
    CUDA_DRIVER_CHECK(cuCtxCreate(&context_, CU_CTX_SCHED_YIELD, device_));
    
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    
    CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    reservation_size_ = ((max_virtual_size + granularity_ - 1) / granularity_) * granularity_;
    CUDA_DRIVER_CHECK(cuMemAddressReserve(&virtual_base_ptr_, reservation_size_, 0, 0, 0));
}

VMMVirtualArena::~VMMVirtualArena() {
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(context_));
    for (auto handle : physical_handles_) {
        cuMemRelease(handle);
    }
    cuMemAddressFree(virtual_base_ptr_, reservation_size_);
    cuCtxDestroy(context_);
}

CUmemGenericAllocationHandle VMMVirtualArena::allocate_physical_chunk(size_t size) {
    size_t aligned_size = ((size + granularity_ - 1) / granularity_) * granularity_;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    
    CUmemGenericAllocationHandle handle;
    CUDA_DRIVER_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
    physical_handles_.push_back(handle);
    return handle;
}

void VMMVirtualArena::swap_layer_chunk(size_t virtual_offset, CUmemGenericAllocationHandle handle, size_t size) {
    size_t aligned_size = ((size + granularity_ - 1) / granularity_) * granularity_;
    CUdeviceptr target_ptr = virtual_base_ptr_ + virtual_offset;
    
    // Atomically unmap existing and map new
    CUDA_DRIVER_CHECK(cuMemUnmap(target_ptr, aligned_size));
    CUDA_DRIVER_CHECK(cuMemMap(target_ptr, aligned_size, 0, handle, 0));
    
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_DRIVER_CHECK(cuMemSetAccess(target_ptr, aligned_size, &access_desc, 1));
}

} // namespace hydra
