import os

files = {
'worker/include/model_runner.hpp': '''#pragma once
#include "../../vmm/include/cuda_vmm.hpp"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstddef>

class ModelRunner {
public:
    ModelRunner(VirtualVramManager& vmm);
    ~ModelRunner();

    void forward_layer_gemm(size_t weight_offset, half* d_input, half* d_output, int m, int n, int k, cudaStream_t stream);

private:
    VirtualVramManager& vmm_;
    cublasLtHandle_t cublas_handle_;
};
''',

'worker/src/model_runner.cpp': '''#include "../include/model_runner.hpp"

ModelRunner::ModelRunner(VirtualVramManager& vmm) : vmm_(vmm) {
    cublasLtCreate(&cublas_handle_);
}

ModelRunner::~ModelRunner() {
    cublasLtDestroy(cublas_handle_);
}

void ModelRunner::forward_layer_gemm(size_t weight_offset, half* d_input, half* d_output, int m, int n, int k, cudaStream_t stream) {
    // truyen thang base va cong offset thay vi cudaMalloc
    half* d_weight = reinterpret_cast<half*>(vmm_.get_base_va() + weight_offset);
    half alpha = 1.0f;
    half beta  = 0.0f;

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    cublasLtMatrixLayout_t adesc, bdesc, cdesc;
    cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16F, k, n, k);
    cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16F, n, m, n);

    cublasLtMatmul(cublas_handle_, operationDesc, &alpha, d_weight, bdesc, d_input, adesc, &beta, d_output, cdesc, d_output, cdesc, NULL, NULL, 0, stream);

    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
''',

'deploy/k8s/hydra-daemonset.yaml': '''apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: hydra-shadow-worker
  namespace: hydra-system
spec:
  selector:
    matchLabels:
      app: hydra-worker
  template:
    metadata:
      labels:
        app: hydra-worker
    spec:
      hostIPC: true
      hostNetwork: true
      containers:
      - name: worker
        image: hydra-worker:latest
        securityContext:
          privileged: true
        resources:
          limits:
            nvidia.com/gpu: "8"
        volumeMounts:
        - name: shared-nvme
          mountPath: /mnt/nvme-of
          readOnly: true
        - name: dev-mem
          mountPath: /dev/mem
      volumes:
      - name: shared-nvme
        hostPath:
          path: /mnt/shared_nvme
          type: DirectoryOrCreate
      - name: dev-mem
        hostPath:
          path: /dev/mem
''',

'control-plane/hydra_controller.go': '''package main

import (
	"fmt"
	"time"
)

type HydraModel struct {
	Name        string
	ModelPath   string
	TotalLayers int
}

type GlobalPageTable struct {
	NodeVRAM map[string]uint64
}

func main() {
	pageTable := &GlobalPageTable{NodeVRAM: make(map[string]uint64)}
	for {
		event := fetchNextHydraModelEvent()
		if event != nil {
			handleModelDeployment(event, pageTable)
		}
		time.Sleep(2 * time.Second)
	}
}

func handleModelDeployment(model *HydraModel, table *GlobalPageTable) {
	targetNodeIP := findNodeForWarmup(table)
	// bypass k8s pod lifecycle, goi thang xuong node de map vram
	sendWarmupRPCToWorker(targetNodeIP, model)
}

func findNodeForWarmup(table *GlobalPageTable) string {
	return "192.168.1.100" 
}

func sendWarmupRPCToWorker(nodeIP string, model *HydraModel) {
	fmt.Printf("[RPC -> %s] warmup %s\\n", nodeIP, model.Name)
}

func fetchNextHydraModelEvent() *HydraModel {
	return &HydraModel{Name: "LLaMA-3-8B", ModelPath: "/mnt/nvme-of/models/llama3-8b.raw", TotalLayers: 32}
}
''',

'ingress/src/gateway.py': '''from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()
request_buffer = asyncio.Queue()

async def check_vram_availability(model_id: str):
    await asyncio.sleep(0.001)
    return "192.168.1.100"

async def dispatch_to_node(node_ip: str, request_data: dict):
    await asyncio.sleep(0.05) 
    return {"token": "Hello"}

@app.post("/generate/{model_id}")
async def generate(model_id: str, prompt: str):
    # dua request vao queue thay vi drop neu vram day
    await request_buffer.put({"model_id": model_id, "prompt": prompt})
    current_req = await request_buffer.get()
    
    target_node = await check_vram_availability(model_id)
    if not target_node:
        raise HTTPException(status_code=503, detail="VRAM full")
        
    result = await dispatch_to_node(target_node, current_req)
    return {"result": result}
''',

'engine/kernels/tensor_table.cuh': '''#pragma once
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
''',

'engine/include/graph_runner.cuh': '''#pragma once
#include <cuda_runtime.h>

class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    void capture_graph(void* virtual_base_addr, int num_layers);
    void launch(cudaStream_t stream);

private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool is_captured_;
};
''',

'engine/src/graph_runner.cu': '''#include "../include/graph_runner.cuh"
#include "../kernels/tensor_table.cuh"

__constant__ LayerPointers c_model_offsets[MAX_LAYERS];
__constant__ void* c_virtual_base_addr;

CudaGraphRunner::CudaGraphRunner() : is_captured_(false) {}

CudaGraphRunner::~CudaGraphRunner() {
    if (is_captured_) {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(graph_);
    }
}

__global__ void dummy_layer_compute(int layer_id) {
    half* qkv_ptr = get_qkv_ptr(layer_id);
}

void CudaGraphRunner::capture_graph(void* virtual_base_addr, int num_layers) {
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < num_layers; ++i) {
        dummy_layer_compute<<<1, 256, 0, capture_stream>>>(i);
    }

    cudaStreamEndCapture(capture_stream, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);
    is_captured_ = true;
    cudaStreamDestroy(capture_stream);
}

void CudaGraphRunner::launch(cudaStream_t stream) {
    if (is_captured_) {
        // goi 1 lan tu driver, khong overhead CPU
        cudaGraphLaunch(graph_exec_, stream);
    }
}
''',

'vmm/include/paged_kv_cache.hpp': '''#pragma once
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
''',

'engine/src/layer_overlap.cu': '''#include <cuda/barrier>
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
'''
}

for filepath, content in files.items():
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
