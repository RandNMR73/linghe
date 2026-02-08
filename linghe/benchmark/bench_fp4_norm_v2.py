# -*- coding: utf-8 -*-
"""
V2 Benchmark for Fused FP4 RMSNorm + Quantization Kernel
Includes L2 Cache Flushing for realistic HBM-bound performance measurement.

Run:
    PYTHONPATH=.:../FastVideo-Quantization python linghe/benchmark/bench_fp4_norm_v2.py
"""

import sys
import time
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
from linghe.utils.fp4_norm import triton_rms_norm_and_fp4_quant_forward

try:
    from flashinfer import nvfp4_quantize, SfLayout
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("WARNING: FlashInfer not installed, cannot run Baseline")


def flush_l2_cache(cache_buffer):
    """
    Flush L2 cache by reading/writing to a large buffer.
    RTX 5090 has ~96MB L2. A 256MB buffer is sufficient.
    """
    cache_buffer.zero_()


def benchmark_with_flush(fn, buffer, n_warmup=10, n_repeat=100):
    """
    Benchmark a function with L2 cache flushing between iterations.
    """
    # Warmup (no flush, just to compile/optimize)
    for _ in range(n_warmup):
        fn()
    
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    
    for i in range(n_repeat):
        # Flush Cache
        flush_l2_cache(buffer)
        torch.cuda.synchronize()
        
        start_events[i].record()
        fn()
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    # Return median time in ms
    import statistics
    return statistics.median(times)


def run_benchmark_v2():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    print(f"Running on {torch.cuda.get_device_name()}")
    
    # Allocate 256MB buffer for cache flushing (float32 = 4 bytes)
    # 256 * 1024 * 1024 / 4 = 67M elements
    cache_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=device)

    configs = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 2048),
        (16384, 5120),
        (32768, 1536),
        (32768, 8192),
    ]

    print("\n|    Rows (M) | Hidden (N) | Baseline (ms) | Fused (ms) | Speedup |")
    print("|-------------|------------|---------------|------------|---------|")

    for M, N in configs:
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        weight = torch.randn(N, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # Baseline
        if HAS_FLASHINFER:
            def baseline_fn():
                rms_sq = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
                rms = torch.rsqrt(rms_sq + eps)
                x_normed = (x.float() * rms * weight.float()).to(torch.bfloat16)
                global_max = x_normed.float().abs().max()
                global_sf = (448.0 * 6.0) / torch.clamp(global_max, min=1e-8)
                out, scale = nvfp4_quantize(x_normed, global_sf.reshape(1), sfLayout=SfLayout.layout_128x4, do_shuffle=False)
                return out, scale
            
            baseline_ms = benchmark_with_flush(baseline_fn, cache_buffer)
        else:
            baseline_ms = float('nan')

        # Fused
        def fused_fn():
            out, scale, _, _ = triton_rms_norm_and_fp4_quant_forward(x, weight, eps=eps)
            return out, scale

        fused_ms = benchmark_with_flush(fused_fn, cache_buffer)
        
        speedup = baseline_ms / fused_ms if baseline_ms > 0 else 0.0
        print(f"| {M:11d} | {N:10d} | {baseline_ms:13.4f} | {fused_ms:10.4f} | {speedup:7.2f}x |")


if __name__ == "__main__":
    run_benchmark_v2()
