#include "cuda_vmm.hpp"
#include <iostream>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        throw std::runtime_error(std::string("CUDA Driver Error: ") + errStr); \
    } \
} while (0)

VirtualVramManager::VirtualVramManager(size_t reserved_size, size_t chunk_size)
    : reserved_size_(reserved_size), chunk_size_(chunk_size), active_bytes_(0), base_va_(0) {
    
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    // 1. Dự phòng dải Virtual Address Space lớn (không tốn physical VRAM)
    CHECK_CU(cuMemAddressReserve(&base_va_, reserved_size_, 0, 0, 0));

    // Cấu hình thuộc tính cấp phát physical backing store
    prop_ = {};
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = dev;

    access_desc_ = {};
    access_desc_.location = prop_.location;
    access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

void* VirtualVramManager::map_backing_store(size_t size) {
    size_t aligned_size = ((size + chunk_size_ - 1) / chunk_size_) * chunk_size_;
    size_t num_chunks = aligned_size / chunk_size_;

    for (size_t i = 0; i < num_chunks; ++i) {
        CUmemGenericAllocationHandle handle;
        // 2. Tạo physical allocation
        CHECK_CU(cuMemCreate(&handle, chunk_size_, &prop_, 0));
        
        // 3. Map physical chunk vào virtual address
        CUdeviceptr target_va = base_va_ + active_bytes_;
        CHECK_CU(cuMemMap(target_va, chunk_size_, 0, handle, 0));
        
        // 4. Cấp quyền Read/Write
        CHECK_CU(cuMemSetAccess(target_va, chunk_size_, &access_desc_, 1));

        allocation_handles_.push_back(handle);
        active_bytes_ += chunk_size_;
    }
    return reinterpret_cast<void*>(base_va_);
}

void VirtualVramManager::evict_all() {
    if (active_bytes_ == 0) return;

    // Unmap toàn bộ physical pages
    CHECK_CU(cuMemUnmap(base_va_, active_bytes_));

    // Giải phóng physical allocation handles trả VRAM về 0
    for (auto handle : allocation_handles_) {
        CHECK_CU(cuMemRelease(handle));
    }
    allocation_handles_.clear();
    active_bytes_ = 0;
}

VirtualVramManager::~VirtualVramManager() {
    evict_all();
    if (base_va_) {
        cuMemAddressFree(base_va_, reserved_size_);
    }
}
