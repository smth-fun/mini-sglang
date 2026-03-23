"""Fused FusedAddRMSNorm + GEMV kernels.

These kernels eliminate the separate FusedAddRMSNorm kernel by computing the norm
inline within the GEMV. Each block independently computes the RMS norm of (x + residual)
and uses the result for the GEMV computation.

Requirements: BLOCK_M >= M (the entire input vector fits in one tile).
This is satisfied for M=1024 (hidden_size of Qwen3-0.6B).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_gemv_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, y_ptr, residual_out_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused: new_res = x + residual; normed = rmsnorm(new_res); y = normed @ W^T.

    Writes GEMV output to y_ptr and updated residual to residual_out_ptr.
    Only block 0 writes the updated residual (to the separate output buffer).
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    # Load x and residual (each block reads the full vectors)
    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    # Fused add
    new_res = x_vals + res_vals

    # Block 0 writes the updated residual to the separate output buffer
    if pid == 0:
        tl.store(
            residual_out_ptr + cols,
            new_res.to(residual_out_ptr.dtype.element_ty),
            mask=mask_m,
        )

    # RMS norm
    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    # Load norm weights and apply
    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = new_res * rms_inv * nw

    # GEMV: y[rows] = dot(x_normed, W[rows, :])
    w_base = w_ptr + rows * stride_wn
    w_offsets = w_base[:, None] + cols[None, :]
    w_vals = tl.load(
        w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0
    ).to(tl.float32)
    acc = tl.sum(w_vals * x_normed[None, :], axis=1)

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_add_rmsnorm_gemv_int8_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """INT8 variant: loads int8 weights and applies per-row scale after dot product."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    new_res = x_vals + res_vals

    if pid == 0:
        tl.store(
            residual_out_ptr + cols,
            new_res.to(residual_out_ptr.dtype.element_ty),
            mask=mask_m,
        )

    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = new_res * rms_inv * nw

    w_base = w_ptr + rows * stride_wn
    w_offsets = w_base[:, None] + cols[None, :]
    w_vals = tl.load(
        w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0
    ).to(tl.float32)
    acc = tl.sum(w_vals * x_normed[None, :], axis=1)

    # Apply per-row scale
    scale = tl.load(scale_ptr + rows, mask=mask_n)
    acc = acc * scale

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_add_rmsnorm_gemv_silu_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, y_ptr, residual_out_ptr,
    M, N_half, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused: new_res = x + residual; normed = rmsnorm(new_res);
    gate_i = dot(normed, W[i, :]); up_i = dot(normed, W[i+N_half, :]);
    y[i] = silu(gate_i) * up_i.

    Combined FusedAddRMSNorm + gate_up GEMV + silu_and_mul.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N_half

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    # Load x and residual
    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    # Fused add
    new_res = x_vals + res_vals

    # Block 0 writes updated residual
    if pid == 0:
        tl.store(
            residual_out_ptr + cols,
            new_res.to(residual_out_ptr.dtype.element_ty),
            mask=mask_m,
        )

    # RMS norm
    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)
    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = new_res * rms_inv * nw

    # Gate and up projections (fused GEMV + silu_and_mul)
    w_gate_base = w_ptr + rows * stride_wn
    w_up_base = w_ptr + (rows + N_half) * stride_wn

    w_g_offsets = w_gate_base[:, None] + cols[None, :]
    w_g = tl.load(
        w_g_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0
    ).to(tl.float32)
    acc_gate = tl.sum(w_g * x_normed[None, :], axis=1)

    w_u_offsets = w_up_base[:, None] + cols[None, :]
    w_u = tl.load(
        w_u_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0
    ).to(tl.float32)
    acc_up = tl.sum(w_u * x_normed[None, :], axis=1)

    # silu(gate) * up
    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    result = gate_silu * acc_up

    tl.store(y_ptr + rows, result.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_add_rmsnorm_gemv_silu_int8_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N_half, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """INT8 variant: loads int8 gate_up weights, applies per-row scale."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N_half

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    res_vals = tl.load(residual_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    new_res = x_vals + res_vals

    if pid == 0:
        tl.store(
            residual_out_ptr + cols,
            new_res.to(residual_out_ptr.dtype.element_ty),
            mask=mask_m,
        )

    sq_sum = tl.sum(new_res * new_res)
    rms_inv = tl.rsqrt(sq_sum / M + eps)
    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = new_res * rms_inv * nw

    w_gate_base = w_ptr + rows * stride_wn
    w_up_base = w_ptr + (rows + N_half) * stride_wn

    w_g_offsets = w_gate_base[:, None] + cols[None, :]
    w_g = tl.load(
        w_g_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0
    ).to(tl.float32)
    acc_gate = tl.sum(w_g * x_normed[None, :], axis=1)

    w_u_offsets = w_up_base[:, None] + cols[None, :]
    w_u = tl.load(
        w_u_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0
    ).to(tl.float32)
    acc_up = tl.sum(w_u * x_normed[None, :], axis=1)

    # Apply per-row scales (gate and up have separate scale rows)
    scale_gate = tl.load(scale_ptr + rows, mask=mask_n)
    scale_up = tl.load(scale_ptr + rows + N_half, mask=mask_n)
    acc_gate = acc_gate * scale_gate
    acc_up = acc_up * scale_up

    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    result = gate_silu * acc_up

    tl.store(y_ptr + rows, result.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_rmsnorm_gemv_kernel(
    x_ptr, norm_w_ptr, w_ptr, y_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused: normed = rmsnorm(x); y = normed @ W^T.

    Used for the first layer where residual is None.
    x_ptr is NOT modified (the caller handles residual = x separately).
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    # RMS norm
    sq_sum = tl.sum(x_vals * x_vals)
    rms_inv = tl.rsqrt(sq_sum / M + eps)
    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = x_vals * rms_inv * nw

    # GEMV
    w_base = w_ptr + rows * stride_wn
    w_offsets = w_base[:, None] + cols[None, :]
    w_vals = tl.load(
        w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0.0
    ).to(tl.float32)
    acc = tl.sum(w_vals * x_normed[None, :], axis=1)

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_rmsnorm_gemv_int8_kernel(
    x_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """INT8 variant of fused RMSNorm + GEMV for first layer."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    cols = tl.arange(0, BLOCK_M)
    mask_m = cols < M

    x_vals = tl.load(x_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)

    sq_sum = tl.sum(x_vals * x_vals)
    rms_inv = tl.rsqrt(sq_sum / M + eps)
    nw = tl.load(norm_w_ptr + cols, mask=mask_m, other=0.0).to(tl.float32)
    x_normed = x_vals * rms_inv * nw

    w_base = w_ptr + rows * stride_wn
    w_offsets = w_base[:, None] + cols[None, :]
    w_vals = tl.load(
        w_offsets, mask=mask_n[:, None] & mask_m[None, :], other=0
    ).to(tl.float32)
    acc = tl.sum(w_vals * x_normed[None, :], axis=1)

    scale = tl.load(scale_ptr + rows, mask=mask_n)
    acc = acc * scale

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def fused_add_rmsnorm_gemv(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    eps: float,
    residual_out: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused FusedAddRMSNorm + GEMV.

    Args:
        x: input (1, M)
        residual: residual (1, M)
        norm_weight: RMSNorm weight (M,)
        gemv_weight: GEMV weight (N, M)
        eps: RMSNorm epsilon
        residual_out: pre-allocated buffer (1, M) for updated residual

    Returns:
        GEMV output (1, N)
    """
    N, M = gemv_weight.shape
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    BLOCK_M = 1024  # Must be >= M
    assert BLOCK_M >= M, f"BLOCK_M={BLOCK_M} must be >= M={M}"
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if weight_scale is not None:
        _fused_add_rmsnorm_gemv_int8_kernel[grid](
            x.view(-1), residual.view(-1), norm_weight, gemv_weight, weight_scale, y.view(-1), residual_out.view(-1),
            M, N, eps,
            gemv_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    else:
        _fused_add_rmsnorm_gemv_kernel[grid](
            x.view(-1), residual.view(-1), norm_weight, gemv_weight, y.view(-1), residual_out.view(-1),
            M, N, eps,
            gemv_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    return y


def fused_add_rmsnorm_gemv_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float,
    residual_out: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused FusedAddRMSNorm + gate_up GEMV + silu_and_mul.

    Args:
        x: input (1, M)
        residual: residual (1, M)
        norm_weight: RMSNorm weight (M,)
        gate_up_weight: merged gate_up weight (2*N_half, M)
        eps: RMSNorm epsilon
        residual_out: pre-allocated buffer (1, M) for updated residual
        weight_scale: per-row scale (2*N_half,) for INT8 dequantization

    Returns:
        output (1, N_half) = silu(normed @ W_gate^T) * (normed @ W_up^T)
    """
    N_full, M = gate_up_weight.shape
    N_half = N_full // 2
    y = torch.empty(1, N_half, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    BLOCK_M = 1024
    assert BLOCK_M >= M, f"BLOCK_M={BLOCK_M} must be >= M={M}"
    grid = ((N_half + BLOCK_N - 1) // BLOCK_N,)

    if weight_scale is not None:
        _fused_add_rmsnorm_gemv_silu_int8_kernel[grid](
            x.view(-1), residual.view(-1), norm_weight, gate_up_weight, weight_scale, y.view(-1), residual_out.view(-1),
            M, N_half, eps,
            gate_up_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    else:
        _fused_add_rmsnorm_gemv_silu_kernel[grid](
            x.view(-1), residual.view(-1), norm_weight, gate_up_weight, y.view(-1), residual_out.view(-1),
            M, N_half, eps,
            gate_up_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    return y


def fused_rmsnorm_gemv(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    eps: float,
    weight_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RMSNorm + GEMV (for first layer where residual is None).

    Args:
        x: input (1, M)
        norm_weight: RMSNorm weight (M,)
        gemv_weight: GEMV weight (N, M)
        eps: RMSNorm epsilon
        weight_scale: per-row scale (N,) for INT8 dequantization

    Returns:
        GEMV output (1, N)
    """
    N, M = gemv_weight.shape
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    BLOCK_M = 1024
    assert BLOCK_M >= M, f"BLOCK_M={BLOCK_M} must be >= M={M}"
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if weight_scale is not None:
        _fused_rmsnorm_gemv_int8_kernel[grid](
            x.view(-1), norm_weight, gemv_weight, weight_scale, y.view(-1),
            M, N, eps,
            gemv_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    else:
        _fused_rmsnorm_gemv_kernel[grid](
            x.view(-1), norm_weight, gemv_weight, y.view(-1),
            M, N, eps,
            gemv_weight.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    return y
