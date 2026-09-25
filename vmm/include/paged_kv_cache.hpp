#pragma once
#include <cuda.h>
#include <vector>
#include <unordered_map>

// 1. KV-Cache Virtual Paging (PagedAttention cấp thấp)
struct LogicalBlock {
    int block_id;       // ID Khối (ví dụ 16/32 tokens)
    CUdeviceptr physical_ptr; 
};

class PagedKVCacheManager {
public:
    PagedKVCacheManager(size_t max_blocks, size_t block_size);
    ~PagedKVCacheManager();

    // Cấp phát Logical Block cho 1 context (tenant) theo cơ chế Paging
    std::vector<LogicalBlock> allocate_blocks(int request_id, int num_blocks);
    
    // Scale-to-zero: Hủy mapping bảng block table, trả về VRAM pool ngay lập tức
    void free_blocks(int request_id);

private:
    size_t block_size_;
    std::vector<bool> free_pool_; // Theo dõi slot vật lý trống
    std::unordered_map<int, std::vector<LogicalBlock>> request_block_table_; // Page Table của Tenant
};
