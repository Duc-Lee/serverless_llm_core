#pragma once
#include <cuda.h>
#include <cstddef>
#include <vector>
#include <stdexcept>

class VirtualVramManager {
public:
    VirtualVramManager(size_t reserved_size, size_t chunk_size);
    ~VirtualVramManager();

    // Map physical memory vào virtual address đã xí trước
    void* map_backing_store(size_t size);
    
    // Thu hồi physical memory đưa VRAM về 0MB, giữ nguyên virtual address reservation
    void evict_all();

    CUdeviceptr get_base_va() const { return base_va_; }
    size_t get_active_bytes() const { return active_bytes_; }

private:
    size_t reserved_size_;
    size_t chunk_size_;
    size_t active_bytes_;
    CUdeviceptr base_va_;
    CUmemAllocationProp prop_;
    CUmemAccessDesc access_desc_;
    std::vector<CUmemGenericAllocationHandle> allocation_handles_;
};
