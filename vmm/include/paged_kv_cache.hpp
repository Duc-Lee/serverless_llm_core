#pragma once
#include <cuda.h>
#include <vector>
#include <unordered_map>

struct LogicalBlock {
    int block_id;
    CUdeviceptr physical_ptr; 
};

class PagedKVCacheManager {
public:
    PagedKVCacheManager(size_t max_blocks, size_t block_size);
    ~PagedKVCacheManager();

    std::vector<LogicalBlock> allocate_blocks(int request_id, int num_blocks);
    
    // xoa mapping khi scale to 0, k free ram
    void free_blocks(int request_id);

private:
    size_t block_size_;
    std::vector<bool> free_pool_;
    std::unordered_map<int, std::vector<LogicalBlock>> request_block_table_;
};
