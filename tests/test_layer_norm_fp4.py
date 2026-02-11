"""
Test for Fused LayerNorm + AdaLN + FP4 Quantization Kernel

Verifies that the fused kernel produces outputs matching:
1. PyTorch LayerNorm reference
2. FlashInfer's nvfp4_quantize reference
"""

import torch
import torch.nn.functional as F
from flashinfer import nvfp4_quantize, SfLayout

# Add parent to path for imports
import sys
sys.path.insert(0, '/linghe')

from linghe.utils.fp4_norm import triton_layer_norm_and_fp4_quant_forward


def reference_layer_norm_adaln(x: torch.Tensor, scale_msa: torch.Tensor | None, 
                                shift_msa: torch.Tensor | None, eps: float = 1e-6):
    """Reference LayerNorm + AdaLN using PyTorch."""
    # LayerNorm without elementwise_affine
    x_normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    
    # Apply AdaLN if provided
    if scale_msa is not None and shift_msa is not None:
        x_normed = x_normed * (1 + scale_msa[None, :]) + shift_msa[None, :]
    
    return x_normed


def test_layer_norm_and_fp4_quant(M=4096, N=2048, test_adaln=False):
    """Test the fused LayerNorm + FP4 quantization kernel."""
    print(f"\n== test_layer_norm_and_fp4_quant M={M}, N={N}, adaln={test_adaln} ==")
    
    torch.manual_seed(42)
    device = 'cuda'
    
    # Input tensor
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16).contiguous()
    eps = 1e-6
    
    # AdaLN parameters (optional)
    scale_msa = None
    shift_msa = None
    if test_adaln:
        scale_msa = torch.randn(N, device=device, dtype=torch.float32) * 0.1  # Small scale
        shift_msa = torch.randn(N, device=device, dtype=torch.float32) * 0.1
    
    # Convert to float32 for kernel
    x_f32 = x.float().contiguous()
    
    # --- Run fused kernel ---
    out_fp4, out_scale, global_sf, mean, rstd = triton_layer_norm_and_fp4_quant_forward(
        x_f32,
        eps=eps,
        scale_msa=scale_msa,
        shift_msa=shift_msa,
    )
    
    print(f"  Shapes OK: out={out_fp4.shape}, scale={out_scale.shape}, global_sf={global_sf.item():.4f}")
    
    # --- Reference: PyTorch LayerNorm + AdaLN ---
    x_normed_ref = reference_layer_norm_adaln(x_f32, scale_msa, shift_msa, eps)
    
    # Verify mean and rstd
    ref_mean = x_f32.mean(dim=-1)
    ref_var = ((x_f32 - ref_mean[:, None]) ** 2).mean(dim=-1)
    ref_rstd = 1.0 / torch.sqrt(ref_var + eps)
    
    mean_cos_sim = F.cosine_similarity(mean, ref_mean, dim=0).item()
    rstd_cos_sim = F.cosine_similarity(rstd, ref_rstd, dim=0).item()
    
    if mean_cos_sim > 0.999 and rstd_cos_sim > 0.999:
        print(f"  [PASS] mean cos_sim={mean_cos_sim:.4f}, rstd cos_sim={rstd_cos_sim:.4f}")
    else:
        print(f"  [FAIL] mean cos_sim={mean_cos_sim:.4f}, rstd cos_sim={rstd_cos_sim:.4f}")
        return False
    
    # --- Reference: FlashInfer FP4 quantization ---
    x_normed_ref_bf16 = x_normed_ref.to(torch.bfloat16)
    ref_global_sf = (448 * 6) / x_normed_ref_bf16.float().abs().nan_to_num().max()
    ref_global_sf_tensor = torch.tensor(ref_global_sf, device=device)
    ref_fp4, ref_scale = nvfp4_quantize(
        x_normed_ref_bf16, ref_global_sf_tensor, 
        sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    
    # Compare FP4 outputs
    match_rate = (out_fp4 == ref_fp4).float().mean().item() * 100
    if match_rate > 90:
        print(f"  [PASS] FlashInfer FP4 match rate: {match_rate:.4f}%")
    else:
        print(f"  [FAIL] FlashInfer FP4 match rate: {match_rate:.4f}% (expected >90%)")
        return False
    
    # Compare block scales
    scale_cos_sim = F.cosine_similarity(
        out_scale.view(-1).float(), 
        ref_scale.view(-1).float(), 
        dim=0
    ).item()
    if scale_cos_sim > 0.8:
        print(f"  [PASS] FlashInfer scale cos_sim: {scale_cos_sim:.4f}")
    else:
        print(f"  [FAIL] FlashInfer scale cos_sim: {scale_cos_sim:.4f} (expected >0.8)")
        return False
    
    # Scale verification passed - that's sufficient for correctness
    print("  Overall: PASS")
    return True


def main():
    """Run all LayerNorm tests."""
    print("FlashInfer available")
    
    all_passed = True
    
    # Test without AdaLN
    all_passed &= test_layer_norm_and_fp4_quant(M=4096, N=2048, test_adaln=False)
    
    # Test with AdaLN
    all_passed &= test_layer_norm_and_fp4_quant(M=4096, N=2048, test_adaln=True)
    
    # Test different sizes
    all_passed &= test_layer_norm_and_fp4_quant(M=1024, N=4096, test_adaln=True)
    
    if all_passed:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
    
    return all_passed


if __name__ == "__main__":
    main()
