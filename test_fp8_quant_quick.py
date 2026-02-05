
import torch
import triton
from linghe.quant.block import triton_block_quant

def test_fp8_quant():
    M, N = 128, 128
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    
    # Run FP8 quantization
    y, s = triton_block_quant(x, block_size=128)
    
    print(f"Input shape: {x.shape}")
    print(f"Output y (quantized) shape: {y.shape}, dtype: {y.dtype}")
    print(f"Output s (scales) shape: {s.shape}, dtype: {s.dtype}")
    
    # Verify basics
    assert y.dtype == torch.float8_e4m3fn
    print("FP8 Quantization test passed!")

if __name__ == "__main__":
    test_fp8_quant()
