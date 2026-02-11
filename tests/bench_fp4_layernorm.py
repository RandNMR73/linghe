"""
Benchmark: Fused FP4LayerNorm vs Original Torch (LayerNorm + AdaLN + nvfp4_quantize)
"""
import torch
import torch.nn.functional as F
import sys
import time

sys.path.insert(0, "/FastVideo-Quantization")

from fastvideo.layers.quantization.fp4_rmsnorm import FP4LayerNorm, triton_layer_norm_and_fp4_quant_forward
from flashinfer import nvfp4_quantize, SfLayout


def benchmark_fn(fn, warmup=50, iters=200):
    """Benchmark a function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration


def original_torch_path(x, norm, scale_msa, shift_msa):
    """Original path: LayerNorm + AdaLN + nvfp4_quantize (called 3x for Q/K/V)."""
    norm_out = (norm(x.float()) * (1 + scale_msa) + shift_msa).to(torch.bfloat16)
    x_flat = norm_out.view(-1, norm_out.shape[-1]).contiguous()
    global_sf = (448 * 6) / x_flat.float().abs().nan_to_num().max()
    x_fp4, x_scale = nvfp4_quantize(x_flat, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    return x_fp4, x_scale, global_sf


def fused_kernel_path(x_flat, fp4_norm, scale_flat, shift_flat):
    """Fused path: FP4LayerNorm (LayerNorm + AdaLN + FP4 quant in one kernel)."""
    return fp4_norm(x_flat, scale_flat, shift_flat)


def main():
    torch.manual_seed(42)
    device = "cuda"

    configs = [
        # (batch*seq, hidden_dim) - real model sizes
        {"name": "WanVideo 1.3B (B=1, S=32760)", "M": 32760, "N": 1536},
        {"name": "Small test (B=1, S=4096)", "M": 4096, "N": 1536},
        {"name": "Medium test (B=1, S=16384)", "M": 16384, "N": 1536},
    ]

    print("=" * 80)
    print("Benchmark: Fused FP4LayerNorm vs Original Torch Path")
    print("  Original = LayerNorm + AdaLN + nvfp4_quantize (called 3x for Q/K/V)")
    print("  Fused    = FP4LayerNorm kernel (1 call, reuse output for Q/K/V)")
    print("=" * 80)

    for cfg in configs:
        M, N = cfg["M"], cfg["N"]
        print(f"\n--- {cfg['name']} ---")
        print(f"  Shape: ({M}, {N})")

        # Create inputs
        x_3d = torch.randn(1, M, N, device=device, dtype=torch.bfloat16)
        scale_msa = torch.randn(1, 1, N, device=device, dtype=torch.float32) * 0.5
        shift_msa = torch.randn(1, 1, N, device=device, dtype=torch.float32) * 0.5
        norm = torch.nn.LayerNorm(N, elementwise_affine=False).to(device)

        # Fused kernel inputs
        x_flat = x_3d.view(-1, N).float().contiguous()
        scale_flat = scale_msa[0, 0, :].contiguous()
        shift_flat = shift_msa[0, 0, :].contiguous()
        fp4_norm = FP4LayerNorm(N, eps=1e-5).to(device)

        # Benchmark original (3 separate quantizations for Q/K/V)
        t_orig_3x = benchmark_fn(
            lambda: (
                original_torch_path(x_3d, norm, scale_msa, shift_msa),
                original_torch_path(x_3d, norm, scale_msa, shift_msa),
                original_torch_path(x_3d, norm, scale_msa, shift_msa),
            )
        )

        # Benchmark original (1 quantization, reuse for Q/K/V - "optimal baseline")
        t_orig_1x = benchmark_fn(
            lambda: original_torch_path(x_3d, norm, scale_msa, shift_msa)
        )

        # Benchmark fused kernel (1 call)
        t_fused = benchmark_fn(
            lambda: fused_kernel_path(x_flat, fp4_norm, scale_flat, shift_flat)
        )

        print(f"  Original (3x quant for Q/K/V): {t_orig_3x:.3f} ms")
        print(f"  Original (1x quant, reuse):    {t_orig_1x:.3f} ms")
        print(f"  Fused kernel:                  {t_fused:.3f} ms")
        print(f"  Speedup vs 3x original:        {t_orig_3x / t_fused:.2f}x")
        print(f"  Speedup vs 1x original:        {t_orig_1x / t_fused:.2f}x")


if __name__ == "__main__":
    main()
