#!/bin/bash
set -e

echo "=== 1. Starting Hydra Shadow Worker Daemon ==="
./build/hydra_worker &
WORKER_PID=$!
sleep 1

echo "=== 2. Starting Control Plane Orchestrator ==="
python3 control-plane/orchestrator.py &
CP_PID=$!
sleep 2

echo "=== 3. Baseline GPU Footprint (Expecting ~0MB VRAM used) ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== 4. Sending Request 1 (Cold-Start Execution) ==="
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "llama-8b", "prompt": "Hello world"}'
echo ""
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== 5. Waiting for 6 seconds (Auto-Scale to Zero) ==="
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Clean up
kill $WORKER_PID $CP_PID
