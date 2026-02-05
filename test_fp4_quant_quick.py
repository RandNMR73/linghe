
import torch
import triton
from linghe.quant.block import triton_block_quant_fp4

def test_fp4_quant():
    M, N = 128, 128
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    
    # Run FP4 quantization
    y, s = triton_block_quant_fp4(x, block_size=128)
    
    print(f"Input shape: {x.shape}")
    print(f"Output y (packed) shape: {y.shape}, dtype: {y.dtype}")
    print(f"Output s (scales) shape: {s.shape}, dtype: {s.dtype}")
    
    assert y.shape == (M, N // 2)
    assert s.shape == (1, 1)
    assert y.dtype == torch.uint8
    
    print("FP4 Quantization test passed!")

if __name__ == "__main__":
    test_fp4_quant()
