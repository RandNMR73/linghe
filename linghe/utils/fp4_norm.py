# -*- coding: utf-8 -*-
"""
FP4 (NVFP4) Fused RMSNorm + Quantization Kernels

Adapted from linghe/utils/norm.py for FP4 format.
Uses FP4 quantization logic DIRECTLY from FastVideo-Quantization/nvfp4_utils.py
to ensure 100% compatibility with mm_fp4 while maintaining kernel fusion.

Copyright (c) Ant Financial Service Group and its affiliates.
"""

import sys
from pathlib import Path

# Add FastVideo-Quantization to path for nvfp4_utils import
_fastvideo_root = Path(__file__).resolve().parent.parent.parent.parent / "FastVideo-Quantization"
if str(_fastvideo_root) not in sys.path:
    sys.path.insert(0, str(_fastvideo_root))

from typing import Optional

import torch
import triton
import triton.language as tl

# Import the FP4 quantization logic from nvfp4_utils (to be inlined)
from nvfp4_utils import _compute_quant_and_scale


@triton.jit
def rms_norm_and_fp4_quant_fused_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,           # Packed FP4 output (uint8)
    scale_ptr,         # Block scales (float8e4nv)
    rms_ptr,           # RMS values for recompute (float32)
    global_sf,         # Single global scale factor (passed as scalar)
    eps,
    M,
    n,
    N: tl.constexpr,   # Next power of 2 of n
    T: tl.constexpr,   # Tiles per program
    W: tl.constexpr,   # Rows per tile
):
    """
    Fused RMSNorm + FP4 Quantization kernel.
    
    1. Computes RMSNorm on-chip.
    2. Calls _compute_quant_and_scale (inlined) on normalized data.
    3. Writes packed output directly to HBM.
    """
    pid = tl.program_id(axis=0)

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = N // 16  # Number of 16-element blocks

    mask = tl.arange(0, N) < n
    weight = tl.load(weight_ptr + tl.arange(0, N), mask=mask).to(tl.float32)[None, :]
    
    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[None, :]
    
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        masks = (indices[:, None] < M) & (tl.arange(0, N) < n)
        
        # Load input
        x = tl.load(x_ptr + offs, mask=masks).to(tl.float32)
        
        # Step 1: RMSNorm
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)
        
        x_normed = x * rms[:, None] * weight
        
        out_tensor, dequant_scale, s_dec = _compute_quant_and_scale(
            x_normed * global_sf,  # Pre-scaled input
            masks,
            use_global_sf=False,    # Don't re-compute global SF
            two_level_quant_P=False
        )
        
        # Store packed FP4 output (half the elements due to packing)
        out_offs = pid * W * T * (n // 2) + i * W * (n // 2) + tl.arange(0, W)[:, None] * (n // 2) + tl.arange(0, N // 2)[None, :]
        out_masks = (indices[:, None] < M) & (tl.arange(0, N // 2) < n // 2)
        tl.store(out_ptr + out_offs, out_tensor, mask=out_masks)
        
        # Store block scales in FlashInfer's expected layout
        NB: tl.constexpr = N // 16
        
        dequant_scale = tl.reshape(dequant_scale, [W, NB])
        
        tl.store(
            scale_ptr + indices[:, None] * (N // 16) + tl.arange(0, NB)[None, :],
            dequant_scale,
            mask=(indices[:, None] < M) & (tl.arange(0, NB)[None, :] < n // 16)
        )
        
        offs += n * W


def triton_rms_norm_and_fp4_quant_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    rms: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + FP4 Quantization forward pass.
    
    100% compatible with mm_fp4.
    Fuses RMSNorm and call to nvfp4_utils._compute_quant_and_scale.
    """
    assert x.is_contiguous() and weight.is_contiguous()
    M, n = x.shape
    N = triton.next_power_of_2(n)
    
    device = x.device
    
    # 1. Compute Global SF (requires 1 pass over data, or can be fused if memory allows)
    # We do a quick RMSNorm pass to get the max.
    # Ideally this would be done in a single kernel reduce, but for now this is efficient enough compared to writing full output.
    x_float = x.float()
    rms_vals = torch.rsqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + eps)
    # Note: We compute max of (x * rms * weight) without saving the tensor
    # We can use a kernel for this or just jit compile it?
    # For simplicity, we compute it.
    x_normed = (x_float * rms_vals * weight.float())
    global_max = x_normed.abs().max()
    global_max = torch.clamp(global_max, min=1e-8)
    global_sf = (448.0 * 6.0) / global_max
    
    # Allocate outputs
    if out is None:
        out = torch.empty((M, n // 2), device=device, dtype=torch.uint8)
    if scale is None:
        scale = torch.empty((M, n // 16), device=device, dtype=torch.float8_e4m3fn)
    if rms is None:
        rms = torch.empty((M,), device=device, dtype=torch.float32)
    
    # Launch Fused Kernel
    W = 8192 // N
    T = 16 // W
    T = max(T, 1)
    grid = (triton.cdiv(M, T * W),)
    
    rms_norm_and_fp4_quant_fused_kernel[grid](
        x,
        weight,
        out,
        scale,
        rms,
        global_sf.item(),
        eps,
        M,
        n,
        N,
        T,
        W,
        num_stages=3,
        num_warps=4
    )
    
    return out, scale, global_sf, rms.squeeze()
