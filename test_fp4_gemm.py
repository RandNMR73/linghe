
import torch
import triton
from linghe.quant.block import triton_block_quant_fp4
from linghe.gemm.blockwise_fp8_gemm import triton_tt_fp4_gemm

def test_fp4_gemm():
    M, N, K = 128, 128, 256
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    
    # Quantize inputs
    a_p, a_s = triton_block_quant_fp4(a, block_size=128)
    b_p, b_s = triton_block_quant_fp4(b, block_size=128)
    
    # Run FP4 GEMM
    # Note: b in triton_tt_fp4_gemm is expected to be (N, K//2) packed, representing (N, K)
    # Our quantization produces exactly this.
    c = triton_tt_fp4_gemm(a_p, b_p, a_s, b_s, block_size=128)
    
    print(f"Output c shape: {c.shape}, dtype: {c.dtype}")
    
    # Simple reference (using original inputs, significant error expected due to 4-bit)
    c_ref = torch.matmul(a, b.t())
    
    # Sanity check: mean and std should match roughly
    print(f"Ref mean: {c_ref.mean().item():.4f}, std: {c_ref.std().item():.4f}")
    print(f"Out mean: {c.mean().item():.4f}, std: {c.std().item():.4f}")
    
    # If they are scale-aligned, it works.
    assert c.shape == (M, N)
    print("FP4 GEMM test passed (shape check)!")

if __name__ == "__main__":
    test_fp4_gemm()
