#include "shadow_vmm.hpp"

namespace hydra {

ShadowWorkerVMM::ShadowWorkerVMM(int device_id, size_t max_virtual_size) {
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

ShadowWorkerVMM::~ShadowWorkerVMM() {
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(context_));
    for (auto& chunk : physical_chunks_) {
        cuMemRelease(chunk.handle);
    }
    cuMemAddressFree(virtual_base_ptr_, reservation_size_);
    cuCtxDestroy(context_);
}

size_t ShadowWorkerVMM::allocate_physical_chunk(size_t size) {
    size_t aligned_size = ((size + granularity_ - 1) / granularity_) * granularity_;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    
    CUmemGenericAllocationHandle handle;
    CUDA_DRIVER_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
    physical_chunks_.push_back({handle, aligned_size});
    return physical_chunks_.size() - 1;
}

void ShadowWorkerVMM::map_chunk_to_virtual(size_t chunk_id) {
    if (chunk_id >= physical_chunks_.size()) throw std::out_of_range("Invalid chunk ID");
    const auto& chunk = physical_chunks_[chunk_id];
    
    CUDA_DRIVER_CHECK(cuMemMap(virtual_base_ptr_, chunk.size, 0, chunk.handle, 0));
    
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    CUDA_DRIVER_CHECK(cuMemSetAccess(virtual_base_ptr_, chunk.size, &access_desc, 1));
}

void ShadowWorkerVMM::unmap_virtual() {
    CUDA_DRIVER_CHECK(cuMemUnmap(virtual_base_ptr_, reservation_size_));
}

} // namespace hydra
