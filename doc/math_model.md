# Mathematical Model: Compute-I/O Overlap in HydraServerless

To achieve zero-overhead layer swapping in Serverless LLM inference, the computation time of the current layer ($T_{compute}$) must strictly hide the DMA transfer time of the next layer ($T_{dma}$).

## Condition for Perfect Overlap
$$ T_{\text{compute}}(L_{\text{prompt}}, B) \ge T_{\text{dma}}(\text{LayerSize}) $$

Where:
- $L_{\text{prompt}}$: The prompt length (Prefill phase).
- $B$: Batch size.
- $\text{LayerSize}$: Size of a single Transformer layer in bytes (e.g., ~256MB for LLaMA-3-8B in FP16).

## 1. DMA Transfer Time ($T_{dma}$)
Using NVIDIA GPUDirect Storage (GDS) over PCIe Gen4 x16 or Gen5 x16, the theoretical bandwidth ($BW_{pcie}$) is ~32 GB/s to 64 GB/s.
$$ T_{\text{dma}} = \frac{\text{LayerSize}}{BW_{\text{pcie}}} $$
For a 256MB layer at 32 GB/s:
$$ T_{\text{dma}} = \frac{256 \text{ MB}}{32 \text{ GB/s}} = 8 \text{ ms} $$

## 2. Compute Time ($T_{compute}$)
During the Prefill phase, computation is compute-bound (GEMM operations).
$$ T_{\text{compute}} \approx \frac{2 \times L_{\text{prompt}} \times B \times d_{\text{model}}^2}{TFLOPS_{\text{hardware}}} $$

To ensure $T_{\text{compute}} \ge 8 \text{ ms}$ on an NVIDIA A100/H100, the prompt length $L_{\text{prompt}}$ must exceed a critical threshold $L^*$.

## Conclusion
If $L_{\text{prompt}} \ge L^*$, the HydraServerless architecture achieves 100% DMA hiding. Layer-0 and Layer-1 are pinned in VRAM to handle the initial latency, allowing the pipeline to stream remaining layers seamlessly, achieving $< 500\text{MB}$ idle footprint per tenant.
