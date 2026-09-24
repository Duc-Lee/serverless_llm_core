# HydraServerless: Master Research & Development Plan

Đây là tài liệu tổng hợp toàn bộ lộ trình nghiên cứu, mô hình kiến trúc, và các Proof-of-Concept (PoC) cốt lõi cho hệ thống HydraServerless.

---

## Phần 1: Kế hoạch Nghiên cứu Tổng thể (Master Plan)

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
