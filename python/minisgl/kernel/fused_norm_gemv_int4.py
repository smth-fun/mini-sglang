"""Fused FusedAddRMSNorm + INT4 GEMV kernels.

INT4 per-group variants of the fused norm+GEMV kernels.
Weights packed as 2 int4 per uint8, with per-group float16 scales.

Two-phase design:
  Phase 1: Compute RMS norm from contiguous load (low register pressure).
  Phase 2: Loop over groups, re-loading even/odd x per group from L1 cache,
           computing normed values, and accumulating INT4 GEMV with scalar scale.
This keeps peak register usage ~3KB vs ~18KB in the single-shot approach.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_gemv_int4_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused: new_res = x + residual; normed = rmsnorm(new_res); y = normed @ dequant(W_int4)^T."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    # Phase 1: Contiguous load for RMS norm
    cols = tl.arange(0, BLOCK_M)
    x_vals = tl.load(x_ptr + cols).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols).to(tl.float32)
    new_res = x_vals + res_vals

    if pid == 0:
        tl.store(residual_out_ptr + cols, new_res.to(residual_out_ptr.dtype.element_ty))

    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    # Phase 2: Group-loop INT4 GEMV
    HALF_GROUP: tl.constexpr = GROUP_SIZE // 2
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = w_ptr + rows * stride_wn

    for g in tl.range(0, NUM_GROUPS):
        col_start = g * GROUP_SIZE
        half_start = col_start // 2
        half_ids = tl.arange(0, HALF_GROUP)

        # Re-load even/odd x, residual, norm_w for this group (from L1 cache)
        x_even = tl.load(x_ptr + col_start + half_ids * 2).to(tl.float32)
        x_odd = tl.load(x_ptr + col_start + half_ids * 2 + 1).to(tl.float32)
        r_even = tl.load(residual_ptr + col_start + half_ids * 2).to(tl.float32)
        r_odd = tl.load(residual_ptr + col_start + half_ids * 2 + 1).to(tl.float32)
        nw_even = tl.load(norm_w_ptr + col_start + half_ids * 2).to(tl.float32)
        nw_odd = tl.load(norm_w_ptr + col_start + half_ids * 2 + 1).to(tl.float32)

        xn_even = (x_even + r_even) * rms_inv * nw_even
        xn_odd = (x_odd + r_odd) * rms_inv * nw_odd

        # Load packed weights for this group
        w_packed = tl.load(
            w_base[:, None] + half_start + half_ids[None, :],
            mask=mask_n[:, None], other=0,
        )
        w_lo = (w_packed & 0x0F).to(tl.float32) - 8.0
        w_hi = (w_packed >> 4).to(tl.float32) - 8.0

        prod = w_lo * xn_even[None, :] + w_hi * xn_odd[None, :]

        # Scalar scale per row per group
        scale = tl.load(scale_ptr + rows * NUM_GROUPS + g, mask=mask_n).to(tl.float32)
        acc += tl.sum(prod, axis=1) * scale

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_add_rmsnorm_gemv_silu_int4_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N_half, eps: tl.constexpr,
    stride_wn,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused add+rmsnorm+gate_up INT4 GEMV+silu."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N_half

    # Phase 1: Contiguous load for RMS norm
    cols = tl.arange(0, BLOCK_M)
    x_vals = tl.load(x_ptr + cols).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols).to(tl.float32)
    new_res = x_vals + res_vals

    if pid == 0:
        tl.store(residual_out_ptr + cols, new_res.to(residual_out_ptr.dtype.element_ty))

    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    # Phase 2: Group-loop INT4 GEMV for gate and up projections
    HALF_GROUP: tl.constexpr = GROUP_SIZE // 2
    acc_gate = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_gate_base = w_ptr + rows * stride_wn
    w_up_base = w_ptr + (rows + N_half) * stride_wn

    for g in tl.range(0, NUM_GROUPS):
        col_start = g * GROUP_SIZE
        half_start = col_start // 2
        half_ids = tl.arange(0, HALF_GROUP)

        # Re-load even/odd values for this group (from L1 cache)
        x_even = tl.load(x_ptr + col_start + half_ids * 2).to(tl.float32)
        x_odd = tl.load(x_ptr + col_start + half_ids * 2 + 1).to(tl.float32)
        r_even = tl.load(residual_ptr + col_start + half_ids * 2).to(tl.float32)
        r_odd = tl.load(residual_ptr + col_start + half_ids * 2 + 1).to(tl.float32)
        nw_even = tl.load(norm_w_ptr + col_start + half_ids * 2).to(tl.float32)
        nw_odd = tl.load(norm_w_ptr + col_start + half_ids * 2 + 1).to(tl.float32)

        xn_even = (x_even + r_even) * rms_inv * nw_even
        xn_odd = (x_odd + r_odd) * rms_inv * nw_odd

        # Gate weights
        wg_packed = tl.load(
            w_gate_base[:, None] + half_start + half_ids[None, :],
            mask=mask_n[:, None], other=0,
        )
        wg_lo = (wg_packed & 0x0F).to(tl.float32) - 8.0
        wg_hi = (wg_packed >> 4).to(tl.float32) - 8.0
        scale_g = tl.load(scale_ptr + rows * NUM_GROUPS + g, mask=mask_n).to(tl.float32)
        acc_gate += tl.sum(wg_lo * xn_even[None, :] + wg_hi * xn_odd[None, :], axis=1) * scale_g

        # Up weights
        wu_packed = tl.load(
            w_up_base[:, None] + half_start + half_ids[None, :],
            mask=mask_n[:, None], other=0,
        )
        wu_lo = (wu_packed & 0x0F).to(tl.float32) - 8.0
        wu_hi = (wu_packed >> 4).to(tl.float32) - 8.0
        scale_u = tl.load(scale_ptr + (rows + N_half) * NUM_GROUPS + g, mask=mask_n).to(tl.float32)
        acc_up += tl.sum(wu_lo * xn_even[None, :] + wu_hi * xn_odd[None, :], axis=1) * scale_u

    # silu(gate) * up
    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    result = gate_silu * acc_up
    tl.store(y_ptr + rows, result.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_rmsnorm_gemv_int4_kernel(
    x_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused rmsnorm+INT4 GEMV for first layer (no residual)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    # Phase 1: Contiguous load for RMS norm
    cols = tl.arange(0, BLOCK_M)
    x_vals = tl.load(x_ptr + cols).to(tl.float32)
    sq_sum = tl.sum(x_vals * x_vals)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    # Phase 2: Group-loop INT4 GEMV
    HALF_GROUP: tl.constexpr = GROUP_SIZE // 2
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = w_ptr + rows * stride_wn

    for g in tl.range(0, NUM_GROUPS):
        col_start = g * GROUP_SIZE
        half_start = col_start // 2
        half_ids = tl.arange(0, HALF_GROUP)

        x_even = tl.load(x_ptr + col_start + half_ids * 2).to(tl.float32)
        x_odd = tl.load(x_ptr + col_start + half_ids * 2 + 1).to(tl.float32)
        nw_even = tl.load(norm_w_ptr + col_start + half_ids * 2).to(tl.float32)
        nw_odd = tl.load(norm_w_ptr + col_start + half_ids * 2 + 1).to(tl.float32)

        xn_even = x_even * rms_inv * nw_even
        xn_odd = x_odd * rms_inv * nw_odd

        w_packed = tl.load(
            w_base[:, None] + half_start + half_ids[None, :],
            mask=mask_n[:, None], other=0,
        )
        w_lo = (w_packed & 0x0F).to(tl.float32) - 8.0
        w_hi = (w_packed >> 4).to(tl.float32) - 8.0

        prod = w_lo * xn_even[None, :] + w_hi * xn_odd[None, :]
        scale = tl.load(scale_ptr + rows * NUM_GROUPS + g, mask=mask_n).to(tl.float32)
        acc += tl.sum(prod, axis=1) * scale

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


# ============= Python launchers =============

def fused_add_rmsnorm_gemv_int4(
    x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
    gemv_weight_int4: torch.Tensor, gemv_weight_scale: torch.Tensor,
    eps: float, residual_out: torch.Tensor, group_size: int = 128,
) -> torch.Tensor:
    N = gemv_weight_int4.shape[0]
    M = gemv_weight_int4.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)
    num_groups = M // group_size

    # BLOCK_N=8, num_warps=2 for large N (lm_head); BLOCK_N=2, num_warps=4 for smaller
    BLOCK_N = 8 if N >= 4096 else 2
    num_warps = 2 if N >= 4096 else 4
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _fused_add_rmsnorm_gemv_int4_kernel[grid](
        x.view(-1), residual.view(-1), norm_weight,
        gemv_weight_int4, gemv_weight_scale,
        y.view(-1), residual_out.view(-1),
        M, N, eps,
        gemv_weight_int4.stride(0),
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_M=M,
        num_warps=num_warps,
    )
    return y


def fused_add_rmsnorm_gemv_silu_int4(
    x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
    gate_up_weight_int4: torch.Tensor, gate_up_weight_scale: torch.Tensor,
    eps: float, residual_out: torch.Tensor, group_size: int = 128,
) -> torch.Tensor:
    N_full = gate_up_weight_int4.shape[0]
    M = gate_up_weight_int4.shape[1] * 2
    N_half = N_full // 2
    y = torch.empty(1, N_half, device=x.device, dtype=x.dtype)
    num_groups = M // group_size

    BLOCK_N = 2
    grid = ((N_half + BLOCK_N - 1) // BLOCK_N,)

    _fused_add_rmsnorm_gemv_silu_int4_kernel[grid](
        x.view(-1), residual.view(-1), norm_weight,
        gate_up_weight_int4, gate_up_weight_scale,
        y.view(-1), residual_out.view(-1),
        M, N_half, eps,
        gate_up_weight_int4.stride(0),
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_M=M,
    )
    return y


def fused_rmsnorm_gemv_int4(
    x: torch.Tensor, norm_weight: torch.Tensor,
    gemv_weight_int4: torch.Tensor, gemv_weight_scale: torch.Tensor,
    eps: float, group_size: int = 128,
) -> torch.Tensor:
    N = gemv_weight_int4.shape[0]
    M = gemv_weight_int4.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)
    num_groups = M // group_size

    BLOCK_N = 2
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _fused_rmsnorm_gemv_int4_kernel[grid](
        x.view(-1), norm_weight,
        gemv_weight_int4, gemv_weight_scale,
        y.view(-1),
        M, N, eps,
        gemv_weight_int4.stride(0),
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_M=M,
    )
    return y
