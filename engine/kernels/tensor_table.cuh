#pragma once
#include <cuda_fp16.h>

#define MAX_LAYERS 128

struct LayerPointers {
    size_t qkv_weight_offset;
    size_t o_weight_offset;
    size_t gate_up_weight_offset;
    size_t down_weight_offset;
    size_t rms_norm_offset;
};

// luu offset tren constant memory de kernel tu read address
extern __constant__ LayerPointers c_model_offsets[MAX_LAYERS];
extern __constant__ void* c_virtual_base_addr;

__device__ __forceinline__ half* get_qkv_ptr(int layer_id) {
    return (half*)((char*)c_virtual_base_addr + c_model_offsets[layer_id].qkv_weight_offset);
}
