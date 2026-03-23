"""Fused gate_up GEMV + silu_and_mul kernel.

Computes: output = silu(x @ W_gate^T) * (x @ W_up^T)
Where W_gate and W_up are the first and second halves of the merged gate_up weight.
This eliminates the intermediate 6144-element buffer and the separate silu_and_mul kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gemv_silu_and_mul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N_half,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused GEMV + silu_and_mul.

    w_ptr points to merged [W_gate; W_up] of shape (2*N_half, M).
    For each output index i in [0, N_half):
      gate_i = dot(x, W[i, :])
      up_i = dot(x, W[i + N_half, :])
      y[i] = silu(gate_i) * up_i
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N_half

    acc_gate = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_N], dtype=tl.float32)

    w_gate_base = w_ptr + rows * stride_wn
    w_up_base = w_ptr + (rows + N_half) * stride_wn

    for col_start in tl.range(0, M, BLOCK_M):
        cols = col_start + tl.arange(0, BLOCK_M)
        mask_m = cols < M
        x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

        # Load gate weights
        w_g_offsets = w_gate_base[:, None] + cols[None, :]
        w_g = tl.load(w_g_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        acc_gate += tl.sum(w_g * x_vals[None, :], axis=1)

        # Load up weights
        w_u_offsets = w_up_base[:, None] + cols[None, :]
        w_u = tl.load(w_u_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        acc_up += tl.sum(w_u * x_vals[None, :], axis=1)

    # silu(gate) * up
    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    result = gate_silu * acc_up

    tl.store(y_ptr + rows, result.to(y_ptr.dtype.element_ty), mask=mask_n)


def fused_gemv_silu_and_mul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused gate_up GEMV + silu_and_mul.

    Args:
        x: input (1, M)
        weight: merged gate_up weight (2*N_half, M)

    Returns:
        output (1, N_half) = silu(x @ W_gate^T) * (x @ W_up^T)
    """
    N_full, M = weight.shape
    N_half = N_full // 2

    y = torch.empty(1, N_half, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    BLOCK_M = min(1024, M)
    grid = ((N_half + BLOCK_N - 1) // BLOCK_N,)

    _fused_gemv_silu_and_mul_kernel[grid](
        x.view(-1), weight, y.view(-1),
        M, N_half,
        weight.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
    return y
