# Serverless LLM Core (HydraServerless)

Nền tảng suy luận Serverless tức thời (Sub-millisecond Scale-to-Zero) tối ưu hóa chi phí và tài nguyên GPU cho Mô hình Ngôn ngữ Lớn (LLM), được thiết kế để bypass hoàn toàn các giới hạn của Kubernetes và Knative trong môi trường AI.

## 🏗 Sơ Đồ Kiến Trúc Hệ Thống (Hydra Global Control Plane)

```mermaid
graph TD
    %% Tầng Ingress & API
    subgraph "Tầng 3: Ingress / Gateway (Python + eBPF)"
        Client[Client Requests] --> Gateway[FastAPI Gateway\n(ingress/src/gateway.py)]
        Gateway -- "Request Buffer" --> Gateway
    end

    %% Tầng Điều Phối (Control Plane)
    subgraph "Tầng 2: Global Control Plane (Go)"
        Gateway -- "Check VRAM Availability" --> Scheduler[VRAM-Aware Global Scheduler\n(control-plane/hydra_controller.go)]
        Scheduler -- "Quản lý Bảng băm VRAM toàn cụm" --> PageTable[(Global Page Table)]
    end

    %% Tầng Thực Thi & GPU (Data Plane)
    subgraph "Tầng 1: Perpetual Shadow Worker (C++ / CUDA)"
        Scheduler -- "1. RPC Warm-up Layer 0\n(Bypass K8s Pod Lifecycle)" --> DaemonSet[K8s DaemonSet Worker\n(deploy/k8s/hydra-daemonset.yaml)]
        Gateway -- "2. Dispatch Token < 1ms\n(Direct Socket)" --> DaemonSet

        subgraph "GPU Pipeline (engine/ & vmm/)"
            DaemonSet --> VMM[Virtual VRAM Manager\n(cuMemMap / Paged KV-Cache)]
            VMM --> GPUDirect[Zero-Copy NVMe-oF GPUDirect\nĐọc RAW Tensors]
            VMM --> Compute[Compute Orchestration\n(CUDA Graphs & Hardware Barriers)]
            Compute --> Kernels[Direct Kernel Weight Addressing\n(cuBLASLt / FlashAttention)]
        end
    end
```

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

Hệ thống được chia làm 3 module chính. Bạn có thể chạy độc lập từng phần để test hoặc deploy toàn cụm bằng Kubernetes.

### 1. Build C++ GPU Inference Worker (`worker/` & `engine/` & `vmm/`)
Yêu cầu: `CMake >= 3.18`, `CUDA Toolkit >= 11.4` (hỗ trợ Driver API và cuBLASLt).

```bash
mkdir build && cd build
cmake ..
make -j
# Chạy worker trực tiếp
./hydra_worker
```

### 2. Khởi chạy Global Scheduler (`control-plane/`)
Bộ điều phối phân tán viết bằng Go, quản lý VRAM toàn cụm và bypass Kubernetes Pod creation.

```bash
cd control-plane
# Khởi chạy Scheduler lắng nghe CRD
go run hydra_controller.go
```

### 3. Khởi chạy Ingress Gateway (`ingress/`)
FastAPI gateway đóng vai trò Request Buffer, ngăn ngừa rớt request khi cluster VRAM quá tải và điều phối xuống node rảnh trong < 1ms.

```bash
cd ingress/src
# Cài đặt requirements (nếu chưa có)
pip install fastapi uvicorn
# Khởi chạy Gateway
python gateway.py
```

### 4. Deploy toàn cụm lên Kubernetes
Triển khai DaemonSet Worker lên các GPU Node (yêu cầu node gắn sẵn NVMe):

```bash
kubectl apply -f deploy/k8s/hydra-daemonset.yaml
```

## 🧩 Các Thành Phần Công Nghệ Lõi Đã Tích Hợp
- **Direct Tensor Binding:** Dùng `cuMemMap` map vùng nhớ ảo mà không copy dữ liệu qua RAM.
- **KV-Cache Virtual Paging:** Cấp phát động logical blocks cho KV-Cache, chống phân mảnh.
- **CUDA Graph Capture:** `cudaGraphLaunch` triệt tiêu CPU launch overhead.
- **Asymmetric Layer Overlap:** Kết hợp `cuda::memcpy_async` và `cuda::barrier` để Copy Engine và Compute Engine chạy độc lập.
- **Tensor Table Constant Memory:** Zero-overhead pointer resolution trong kernel CUDA.
