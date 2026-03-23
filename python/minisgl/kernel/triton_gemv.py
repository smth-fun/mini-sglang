"""Optimized Triton GEMV kernel for batch size 1 decoding.

For small matrices (hidden_size 1024-3072), this kernel achieves significantly
higher bandwidth utilization than cuBLAS GEMV inside CUDA graphs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """GEMV: y[1,N] = x[1,M] @ W[N,M]^T

    Each program handles BLOCK_N output rows, reducing over M in chunks of BLOCK_M.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = w_ptr + rows * stride_wn

    for col_start in tl.range(0, M, BLOCK_M):
        cols = col_start + tl.arange(0, BLOCK_M)
        mask_m = cols < M
        x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
        w_offsets = w_base[:, None] + cols[None, :]
        w_vals = tl.load(
            w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _gemv_int8_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    M, N,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """INT8 GEMV: y[1,N] = x[1,M] @ (W_int8[N,M] * scale[N])^T

    Same as _gemv_kernel but loads INT8 weights and applies per-row scale.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = w_ptr + rows * stride_wn

    for col_start in tl.range(0, M, BLOCK_M):
        cols = col_start + tl.arange(0, BLOCK_M)
        mask_m = cols < M
        x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
        w_offsets = w_base[:, None] + cols[None, :]
        w_vals = tl.load(
            w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0
        ).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Apply per-row scale after accumulation: result = sum(w_int8 * x) * scale
    scale = tl.load(scale_ptr + rows, mask=mask_n)
    acc = acc * scale

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def _get_config(M: int, N: int):
    """Select BLOCK_N, BLOCK_M based on matrix dimensions.

    Micro-benchmarked inside CUDA graphs: BLOCK_N=2 with BLOCK_M=1024
    consistently gets the highest bandwidth utilization across all sizes.
    """
    BLOCK_N = 2
    BLOCK_M = min(1024, M)
    return BLOCK_N, BLOCK_M


# Size threshold: only use Triton for sizes where it beats cuBLAS
# Based on micro-benchmarks inside CUDA graphs:
# - QKV (N=2048, M=1024): cuBLAS wins → skip
# - O (N=1024, M=1024): Triton 2x faster → use
# - gate_up (N=6144, M=1024): Triton 1.67x faster → use
# - down (N=1024, M=3072): Triton 1.67x faster → use
# - lm_head (N=151936, M=1024): roughly equal → skip
def _should_use_triton(M: int, N: int) -> bool:
    """Decide whether Triton GEMV is faster than cuBLAS for this size."""
    # Skip very large N (lm_head) - cuBLAS is fine
    if N > 10000:
        return False
    # Skip QKV-like sizes where cuBLAS wins (N=2048, M=1024)
    if N == 2048 and M == 1024:
        return False
    return True


def triton_gemv(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None, weight_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for F.linear(x, weight) optimized for bs=1.

    Args:
        x: input tensor of shape (1, M) or (M,)
        weight: weight matrix of shape (N, M), bf16 or int8
        out: optional pre-allocated output tensor of shape (1, N) or (N,)
        weight_scale: per-row scale factors of shape (N,) for INT8 weights

    Returns:
        output tensor of shape matching input batch dims + (N,)
    """
    orig_shape = x.shape
    if x.dim() > 1:
        x_flat = x.view(-1)
        batch_size = x.shape[0] if x.dim() == 2 else 1
    else:
        x_flat = x
        batch_size = 1

    N, M = weight.shape

    # Only use triton GEMV for batch size 1
    if batch_size != 1:
        return torch.nn.functional.linear(x, weight)

    if out is not None:
        y = out.view(-1)
    else:
        y = torch.empty(N, device=x.device, dtype=x.dtype)
    BLOCK_N, BLOCK_M = _get_config(M, N)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if weight_scale is not None:
        _gemv_int8_kernel[grid](
            x_flat, weight, weight_scale, y,
            M, N,
            weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    else:
        _gemv_kernel[grid](
            x_flat, weight, y,
            M, N,
            weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )

    if len(orig_shape) > 1:
        return y.view(1, N)
    return y
