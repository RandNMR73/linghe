"""
Compare scale layout between fused kernel and nvfp4_quantize
"""
import torch
import sys
sys.path.insert(0, '/FastVideo-Quantization')
sys.path.insert(0, '/linghe')

from flashinfer import nvfp4_quantize, SfLayout, mm_fp4
from fastvideo.layers.quantization.fp4_rmsnorm import FP4LayerNorm, triton_layer_norm_and_fp4_quant_forward

def compare_scale_layouts():
    # Same dimensions as real model
    M = 32760
    N = 1536
    
    # Create random input matching real model stats
    torch.manual_seed(42)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 0.1
    x.clamp_(-3, 3)  # Similar to real input range
    
    # AdaLN parameters
    scale_msa = torch.randn(N, device='cuda', dtype=torch.float32) * 0.5
    shift_msa = torch.randn(N, device='cuda', dtype=torch.float32) * 0.5
    
    print(f"Input shape: {x.shape}")
    
    # 1. Reference: LayerNorm + AdaLN + nvfp4_quantize
    x_norm = torch.nn.functional.layer_norm(x, (N,))
    x_modulated = x_norm * (1 + scale_msa) + shift_msa
    
    global_sf_ref = (448 * 6) / x_modulated.abs().max()
    x_fp4_ref, x_scale_ref = nvfp4_quantize(x_modulated.bfloat16(), global_sf_ref, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    
    print(f"\nReference (nvfp4_quantize):")
    print(f"  x_fp4 shape: {x_fp4_ref.shape}, dtype: {x_fp4_ref.dtype}")
    print(f"  x_scale shape: {x_scale_ref.shape}, dtype: {x_scale_ref.dtype}")
    print(f"  global_sf: {global_sf_ref}")
    print(f"  x_scale first row bytes: {x_scale_ref[0, :8].view(torch.int8)}")
    
    # 2. Fused kernel
    x_fp4_fused, x_scale_fused, global_sf_fused, _, _ = triton_layer_norm_and_fp4_quant_forward(
        x.contiguous(), eps=1e-5, scale_msa=scale_msa, shift_msa=shift_msa
    )
    
    print(f"\nFused kernel:")
    print(f"  x_fp4 shape: {x_fp4_fused.shape}, dtype: {x_fp4_fused.dtype}")
    print(f"  x_scale shape: {x_scale_fused.shape}, dtype: {x_scale_fused.dtype}")
    print(f"  global_sf: {global_sf_fused.item()}")
    print(f"  x_scale first row bytes: {x_scale_fused[0, :8].view(torch.int8)}")
    
    # 3. Compare shapes and values
    print(f"\nShape comparison:")
    print(f"  x_fp4: ref={x_fp4_ref.shape} vs fused={x_fp4_fused.shape}")
    print(f"  x_scale: ref={x_scale_ref.shape} vs fused={x_scale_fused.shape}")
    
    # Check if scale layouts match
    scale_match = x_scale_ref.shape == x_scale_fused.shape
    if scale_match:
        # Compare actual bytes
        scale_bytes_ref = x_scale_ref.view(torch.int8)
        scale_bytes_fused = x_scale_fused.view(torch.int8)
        byte_match_rate = (scale_bytes_ref == scale_bytes_fused).float().mean().item()
        print(f"  Scale byte match rate: {byte_match_rate*100:.2f}%")
        
        # Check where they differ
        if byte_match_rate < 1.0:
            diff_mask = scale_bytes_ref != scale_bytes_fused
            diff_indices = diff_mask.nonzero()
            print(f"  First 5 differing positions: {diff_indices[:5].tolist()}")
    else:
        print(f"  SHAPE MISMATCH!")
    
    # 4. Test mm_fp4 with reference scales
    print(f"\n--- Testing mm_fp4 ---")
    weight = torch.randn(N, N, device='cuda', dtype=torch.bfloat16)
    weight_sf = (448 * 6) / weight.float().abs().max()
    weight_fp4, weight_scale = nvfp4_quantize(weight, weight_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    
    # Test with reference
    try:
        out_ref = mm_fp4(x_fp4_ref, weight_fp4.T, x_scale_ref, weight_scale.T, 
                         1.0/(global_sf_ref * weight_sf), torch.bfloat16, None, backend='cutlass')
        nan_ref = torch.isnan(out_ref).any().item()
        print(f"Reference mm_fp4: NaN={nan_ref}")
    except Exception as e:
        print(f"Reference mm_fp4 failed: {e}")
    
    # Test with fused kernel output
    try:
        out_fused = mm_fp4(x_fp4_fused, weight_fp4.T, x_scale_fused, weight_scale.T, 
                          1.0/(global_sf_fused.item() * weight_sf), torch.bfloat16, None, backend='cutlass')
        nan_fused = torch.isnan(out_fused).any().item()
        print(f"Fused kernel mm_fp4: NaN={nan_fused}")
    except Exception as e:
        print(f"Fused kernel mm_fp4 failed: {e}")

if __name__ == "__main__":
    compare_scale_layouts()
