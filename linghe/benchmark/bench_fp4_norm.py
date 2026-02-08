# -*- coding: utf-8 -*-
"""
Benchmark for Fused FP4 RMSNorm + Quantization Kernel

Compariing latencies between:
1. Baseline: PyTorch RMSNorm + FlashInfer nvfp4_quantize (Separated)
2. Fused Kernel: triton_rms_norm_and_fp4_quant_forward (Fused)

Run:
    PYTHONPATH=.:../FastVideo-Quantization python linghe/benchmark/bench_fp4_norm.py
"""

import sys
from pathlib import Path

# Add paths for imports
_linghe_root = Path(__file__).resolve().parent.parent.parent
if str(_linghe_root) not in sys.path:
    sys.path.insert(0, str(_linghe_root))

_fastvideo_root = _linghe_root.parent / "FastVideo-Quantization"
if str(_fastvideo_root) not in sys.path:
    sys.path.insert(0, str(_fastvideo_root))

import torch
import triton
import triton.testing

from linghe.utils.fp4_norm import triton_rms_norm_and_fp4_quant_forward

try:
    from flashinfer import nvfp4_quantize, SfLayout
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("WARNING: FlashInfer not installed, cannot run Baseline")


def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    print(f"Running on {torch.cuda.get_device_name()}")

    configs = [
        # (M, N) - batch*seq_len, hidden_dim
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 2048),
        (16384, 5120),  # Typical WanVideo large activation sizes
        (32768, 1536), #Wan1.3b
        (32768, 8192),
    ]

    results = []

    print("\n|    Rows (M) | Hidden (N) | Baseline (ms) | Fused (ms) | Speedup |")
    print("|-------------|------------|---------------|------------|---------|")

    for M, N in configs:
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        weight = torch.randn(N, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # --- Baseline (RMSNorm + NVFP4 Quantize) ---
        if HAS_FLASHINFER:
            def baseline_fn():
                # Step 1: RMSNorm
                rms_sq = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
                rms = torch.rsqrt(rms_sq + eps)
                x_normed = (x.float() * rms * weight.float()).to(torch.bfloat16)
                
                # Step 2: Global Scale (Manual reduction required by NVFP4 algo)
                global_max = x_normed.float().abs().max()  # This reduction is expensive!
                global_sf = (448.0 * 6.0) / torch.clamp(global_max, min=1e-8)
                
                # Step 3: FlashInfer Quantize
                out, scale = nvfp4_quantize(
                    x_normed, 
                    global_sf.reshape(1),
                    sfLayout=SfLayout.layout_128x4, 
                    do_shuffle=False
                )
                return out, scale
            
            # Use median (0.5) to filter outliers and sufficient warmup
            # baseline_ms = triton.testing.do_bench(baseline_fn, warmup=100, rep=100, quantiles=[0.5])
            baseline_ms = triton.testing.do_bench(baseline_fn, warmup=100, rep=100)
        else:
            baseline_ms = float('nan')

        # --- Fused Implementation ---
        def fused_fn():
            out, scale, _, _ = triton_rms_norm_and_fp4_quant_forward(x, weight, eps=eps)
            return out, scale, None, None # Match baseline_fn signature? No need.

        # fused_ms = triton.testing.do_bench(fused_fn, warmup=100, rep=100, quantiles=[0.5])
        fused_ms = triton.testing.do_bench(fused_fn, warmup=100, rep=100)

        speedup = baseline_ms / fused_ms if baseline_ms > 0 else 0.0

        print(f"| {M:11d} | {N:10d} | {baseline_ms:13.4f} | {fused_ms:10.4f} | {speedup:7.2f}x |")
        results.append((M, N, baseline_ms, fused_ms, speedup))


if __name__ == "__main__":
    run_benchmark()
