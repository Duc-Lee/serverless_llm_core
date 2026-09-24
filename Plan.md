# HydraServerless: Master Research & Development Plan

Đây là tài liệu tổng hợp toàn bộ lộ trình nghiên cứu, mô hình kiến trúc, và các Proof-of-Concept (PoC) cốt lõi cho hệ thống HydraServerless.

---

## Phần 1: Kế hoạch Nghiên cứu Tổng thể (Master Plan)

**Mục tiêu**: OSDI / ASPLOS / SOSP
**Chủ đề**: Sub-Millisecond Scale-to-Zero Serverless LLM Inference

### Giai đoạn 1: Xây dựng Hệ thống Lõi (Core Systems Engineering)
*Mục tiêu: Hoàn thiện 3 trụ cột kỹ thuật, đảm bảo không có overhead ẩn từ OS/Driver.*

- [x] **Trụ cột 1: Shadow CUDA Worker & Virtual Context Swapping**
  - [x] Thiết kế VMM memory reservation (`cuMemAddressReserve`).
  - [x] Viết Prototype C++20 đo độ trễ Context Swap (Đạt < 50us).
  - [ ] Tích hợp VMM vào một inference engine thu nhỏ (ví dụ: một custom forward pass của Llama).
- [ ] **Trụ cột 2: Tiered Weight Slicing & GPUDirect DMA**
  - [ ] Khởi tạo ring-buffer cho Asynchronous DMA.
  - [ ] Tích hợp thư viện `cufile` (GPUDirect Storage) để kéo weights từ NVMe thẳng vào Physical Chunk đã được alloc.
  - [ ] Viết mô-đun Pipeliner: Đồng bộ hóa giữa luồng Compute (Tensor Core) và luồng IO (PCIe DMA).
- [ ] **Trụ cột 3: L7 eBPF Early Prompt Peeker**
  - [ ] Viết eBPF TC/sockops hook bằng C để bắt gói tin TCP (HTTP POST).
  - [ ] Bóc tách JSON payload siêu nhanh trong kernel space để lấy `L_prompt` và `model_id`.
  - [ ] Thiết lập luồng IPC (Out-of-band) bắn tín hiệu từ Kernel eBPF sang Shadow Worker (Userspace).

### Giai đoạn 2: Mô hình Toán học & Chứng minh (Formalization)
*Mục tiêu: Đưa ra nền tảng lý thuyết vững chắc cho ban giám khảo (PC) về tại sao thiết kế này không bị nghẽn (bottleneck).*

- [ ] **Mô hình hóa IO vs. Compute**: 
  - Tính toán công thức: $T_{\text{compute\_layer}(k)} \ge T_{\text{dma\_transfer\_layer}(k+1)}$
  - Tham số hóa: TFLOPS thực tế của A100/H100 so với băng thông PCIe Gen4/Gen5 ($32-64\text{ GB/s}$).
- [ ] **Tối ưu hóa Phân rã Lớp (Layer Slicing)**:
  - Xác định "Điểm gãy" (Break-even point): Giữ bao nhiêu lớp (Layer 0..$N$) trong VRAM (Pinned) là tối ưu.
- [ ] **Phân tích Memory Fragmentation**: 
  - Đánh giá hiện tượng phân mảnh của VMM khi swap hàng nghìn Tenants.

### Giai đoạn 3: Tích hợp Hệ thống (End-to-End Integration)
- [ ] **Control Plane (Go)**: Quản lý metadata, tenant pool, xử lý eBPF signal.
- [ ] **Inference Engine Adapter**: Sửa đổi cơ chế PagedAttention để sử dụng các dải địa chỉ ảo (Virtual Address) do VMM quản lý thay vì `cudaMalloc`.

### Giai đoạn 4: Đánh giá & Thực nghiệm (Benchmarking - OSDI Standard)
- [ ] **Microbenchmarks**: TTFT Latency, Băng thông PCIe utilization graph.
- [ ] **Macrobenchmarks**: Azure/AWS Lambda Traces, Tail Latency.
- [ ] **Economic Analysis**: Cost Efficiency / VRAM savings.

---

## Phần 2: Trụ cột 1 - Shadow CUDA VMM (Proof of Concept)

Đoạn mã C++20 sử dụng CUDA Driver API để chứng minh khả năng hoán đổi Virtual Memory cho Tenant siêu tốc (sub-millisecond) mà không cần khởi tạo lại Context.

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <format>
#include <string>

// CUDA Driver API is strictly required for Low-Level VMM Operations
#include <cuda.h>

#define CUDA_DRIVER_CHECK(call)                                                \
    do {                                                                       \
        CUresult result = call;                                                \
        if (result != CUDA_SUCCESS) {                                          \
            const char* msg;                                                   \
            cuGetErrorString(result, &msg);                                    \
            const char* name;                                                  \
            cuGetErrorName(result, &name);                                     \
            throw std::runtime_error(                                          \
                std::format("CUDA Driver API error: {} ({}) at {}:{}",         \
                            name, msg, __FILE__, __LINE__));                   \
        }                                                                      \
    } while (0)

class ShadowWorkerVMM {
private:
    CUdevice device_;
    CUcontext context_;
    CUdeviceptr virtual_base_ptr_;
    size_t reservation_size_;
    size_t granularity_;

    struct PhysicalChunk {
        CUmemGenericAllocationHandle handle;
        size_t size;
    };
    std::vector<PhysicalChunk> physical_chunks_;

public:
    ShadowWorkerVMM(int device_id, size_t max_virtual_size) {
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
        
        std::cout << "[ShadowWorker] Reserved " << reservation_size_ / (1024.0 * 1024.0) 
                  << " MB of pure Virtual Address space.\n";
    }
    
    ~ShadowWorkerVMM() {
        CUDA_DRIVER_CHECK(cuCtxSetCurrent(context_));
        for (auto& chunk : physical_chunks_) {
            cuMemRelease(chunk.handle);
        }
        cuMemAddressFree(virtual_base_ptr_, reservation_size_);
        cuCtxDestroy(context_);
    }

    size_t allocate_physical_chunk(size_t size) {
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

    void map_chunk_to_virtual(size_t chunk_id) {
        if (chunk_id >= physical_chunks_.size()) throw std::out_of_range("Invalid chunk ID");
        const auto& chunk = physical_chunks_[chunk_id];
        
        CUDA_DRIVER_CHECK(cuMemMap(virtual_base_ptr_, chunk.size, 0, chunk.handle, 0));
        
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        
        CUDA_DRIVER_CHECK(cuMemSetAccess(virtual_base_ptr_, chunk.size, &access_desc, 1));
    }
};

void run_vmm_benchmark() {
    int device_id = 0;
    size_t virtual_arena_size = 16ULL * 1024 * 1024 * 1024; 
    ShadowWorkerVMM worker(device_id, virtual_arena_size);
    
    size_t chunk_size = 2ULL * 1024 * 1024 * 1024;
    size_t chunk0_id = worker.allocate_physical_chunk(chunk_size);
    size_t chunk1_id = worker.allocate_physical_chunk(chunk_size);

    worker.map_chunk_to_virtual(chunk0_id);
    CUDA_DRIVER_CHECK(cuCtxSynchronize());

    const int iterations = 1000;
    double total_us = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        size_t target_chunk = (i % 2 == 0) ? chunk1_id : chunk0_id;
        auto start = std::chrono::high_resolution_clock::now();
        worker.map_chunk_to_virtual(target_chunk);
        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    std::cout << "Actual VMM Context Swap Latency: " << (total_us / iterations) << " us\n";
}

int main() {
    try { run_vmm_benchmark(); } catch (const std::exception& e) { return 1; }
    return 0;
}
```

---

## Phần 3: Trụ cột 2 - Tiered Weight Slicing & Toán học DMA Overlapping

### 1. Mô hình Toán học: Điều kiện Overlap Hoàn hảo
Giả sử:
- $W_k$: Dung lượng bộ nhớ (Bytes) của Lớp $k$.
- $C_k$: Số lượng phép toán (FLOPs) cần thiết để tính toán Lớp $k$.
- $BW_{\text{pcie}}$: Băng thông thực tế của PCIe Gen5.
- $P_{\text{gpu}}$: Công suất tính toán thực tế của GPU.

**Bất phương trình Cốt lõi của HydraServerless:**
$$T_{\text{compute}}(k) \ge T_{\text{dma}}(k+1) + \delta_{\text{overhead}}$$

Để giải quyết sự thiếu hụt Compute Time (do LLM Decoding bị Memory-bound), HydraServerless sử dụng chiến lược **Tiered Slicing**. Ghim $M$ lớp đầu tiên trên VRAM. Bất phương trình trở thành:
$$\sum_{i=0}^{M-1} T_{\text{compute}}(i) \ge T_{\text{dma}}(M)$$

### 2. Kiến trúc Luồng DMA & VMM Ring Buffer
Hệ thống duy trì một Ring Buffer chứa các Physical Chunks (đã alloc bằng `cuMemCreate`).
DMA (thông qua GPUDirect Storage/`io_uring`) kéo weights vào chunk.
GPU `cuMemMap` các chunk này ngay khi DMA kéo xong, tạo ra một dải Virtual Address liên tục cho Tensor Core. Toàn bộ độ trễ copy memory của CPU được loại bỏ.

---

## Phần 4: Trụ cột 3 - Layer-7 eBPF Early Peeker

Mã nguồn C chạy trong Kernel (eBPF) để "nhòm" trước gói tin TCP tại tầng Traffic Control, bỏ qua toàn bộ Network Stack của OS.

```c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

struct inference_trigger_event {
    __u32 saddr;
    __u32 daddr;
    __u16 sport;
    __u16 dport;
    __u32 model_hash;
    __u32 estimated_len;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} trigger_events SEC(".maps");

static __always_inline __u32 fnv1a_32(const char *data, int len) {
    __u32 hash = 2166136261U;
    for (int i = 0; i < len && i < 32; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 16777619U;
    }
    return hash;
}

SEC("tc")
int hydra_early_peeker(struct __sk_buff *skb) {
    void *data_end = (void *)(long)skb->data_end;
    void *data = (void *)(long)skb->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end || eth->h_proto != bpf_htons(ETH_P_IP)) return TC_ACT_OK;

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end || iph->protocol != IPPROTO_TCP) return TC_ACT_OK;

    struct tcphdr *tcph = (void *)iph + (iph->ihl * 4);
    if ((void *)(tcph + 1) > data_end || tcph->dest != bpf_htons(8080)) return TC_ACT_OK;

    __u32 tcp_header_len = tcph->doff * 4;
    unsigned char *payload = (unsigned char *)tcph + tcp_header_len;
    
    if ((void *)(payload + 16) > data_end) return TC_ACT_OK;

    if (payload[0] == 'P' && payload[1] == 'O' && payload[2] == 'S' && payload[3] == 'T') {
        struct inference_trigger_event *event;
        event = bpf_ringbuf_reserve(&trigger_events, sizeof(*event), 0);
        if (!event) return TC_ACT_OK;

        event->saddr = iph->saddr;
        event->daddr = iph->daddr;
        event->sport = bpf_ntohs(tcph->source);
        event->dport = bpf_ntohs(tcph->dest);
        event->estimated_len = bpf_ntohs(iph->tot_len) - (iph->ihl * 4) - tcp_header_len;
        event->model_hash = fnv1a_32("llama-3-8b", 10);

        bpf_ringbuf_submit(event, 0);
    }
    return TC_ACT_OK;
}
char _license[] SEC("license") = "GPL";
```

---

## Phần 5: Kịch bản Đánh giá (Evaluation Benchmarks)

### 1. Môi trường Thực nghiệm (Testbed Setup)
- **Hardware**: NVIDIA H100 80GB PCIe Gen5.
- **Storage**: Local NVMe SSD (Gen5).
- **Baselines**: Knative + vLLM (Cold), ServerlessLLM, vLLM (Warm - Upper bound).

### 2. Microbenchmarks
- **Shadow CUDA VMM Jitter**: Đo CDF của độ trễ `cuMemMap`. Mục tiêu: $>99\%$ < $50\mu s$.
- **DMA vs Compute Pipelining**: Đồ thị băng thông (GB/s). Đường DMA lọt thỏm dưới đường Compute.
- **eBPF Overhead**: Bơm 100K RPS. Overhead mạng tăng < $10\mu s$.

### 3. Macrobenchmarks
- **TTFT Scalability**: Bảng TTFT trên Llama-3 (8B -> 405B). TTFT của Hydra cố định mức 50-100ms.
- **Bursty Workload**: Azure Functions Trace. Đo P99 Tail Latency dưới traffic spike từ 0 -> 1000 RPS.

### 4. Phân tích Kinh tế
- Bằng việc ghim 5% VRAM, một GPU 80GB chứa được 20 Tenants. So sánh mức tiết kiệm VRAM và TCO trên mỗi query so với hệ thống pre-warmed.
