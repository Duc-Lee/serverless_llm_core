#!/bin/bash
# Script cấu hình quyền truy cập DMA và kernel modules (GPUDirect Storage)
echo "[Deploy] Setting up NVIDIA GPUDirect Storage (GDS)..."

# Bật Peer-to-Peer memory cho DMA
modprobe nvidia-peermem
if [ $? -eq 0 ]; then
    echo "[OK] nvidia-peermem loaded."
else
    echo "[ERROR] Failed to load nvidia-peermem."
fi

# Bật cuFile (GPUDirect Storage Driver)
modprobe cufile_drv
if [ $? -eq 0 ]; then
    echo "[OK] cufile_drv loaded."
else
    echo "[ERROR] Failed to load cufile_drv."
fi

echo "[Deploy] GDS Kernel modules are ready!"
