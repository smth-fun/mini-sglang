"""INT4 per-group Triton GEMV kernel for batch size 1 decoding.

Weights are packed as 2 int4 values per uint8 byte with per-group float16 scales.
Packing: byte = ((val0 + 8) & 0xF) | (((val1 + 8) & 0xF) << 4)
Dequant: val0 = (byte & 0xF) - 8, val1 = (byte >> 4) - 8, then multiply by group scale.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_int4_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    M, N,
    stride_wn,      # packed weight stride along N dim (M//2)
    num_groups,      # M // GROUP_SIZE
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """INT4 GEMV: load BLOCK_M elements per iteration (BLOCK_M/2 packed bytes)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    HALF_BLOCK: tl.constexpr = BLOCK_M // 2
    w_base = w_ptr + rows * stride_wn

    for col_start in tl.range(0, M, BLOCK_M):
        # Load packed uint8 weights: (BLOCK_N, HALF_BLOCK)
        packed_start = col_start // 2
        packed_ids = packed_start + tl.arange(0, HALF_BLOCK)
        w_offsets = w_base[:, None] + packed_ids[None, :]
        w_packed = tl.load(w_offsets, mask=mask_n[:, None], other=0)

        # Unpack int4 pairs
        w_lo = (w_packed & 0x0F).to(tl.float32) - 8.0   # even elements
        w_hi = (w_packed >> 4).to(tl.float32) - 8.0      # odd elements

        # Load x with stride-2 (from L2 cache, shared across all blocks)
        half_ids = tl.arange(0, HALF_BLOCK)
        x_even = tl.load(x_ptr + col_start + half_ids * 2).to(tl.float32)
        x_odd = tl.load(x_ptr + col_start + half_ids * 2 + 1).to(tl.float32)

        # Element-wise products (before scaling)
        prod = w_lo * x_even[None, :] + w_hi * x_odd[None, :]  # (BLOCK_N, HALF_BLOCK)

        # Load per-group scales expanded to per-pair via gather (hits L1 cache)
        group_start = col_start // GROUP_SIZE
        pair_group_ids = (half_ids * 2) // GROUP_SIZE  # (HALF_BLOCK,) - which group each pair belongs to
        scales = tl.load(
            scale_ptr + rows[:, None] * num_groups + (group_start + pair_group_ids[None, :]),
            mask=mask_n[:, None], other=1.0,
        ).to(tl.float32)  # (BLOCK_N, HALF_BLOCK)

        acc += tl.sum(prod * scales, axis=1)

    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def triton_gemv_int4(
    x: torch.Tensor,
    weight_int4: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """INT4 GEMV for bs=1 decode."""
    orig_shape = x.shape
    x_flat = x.view(-1)
    N = weight_int4.shape[0]
    M = weight_int4.shape[1] * 2

    y = torch.empty(N, device=x.device, dtype=x.dtype)
    num_groups = M // group_size

    # BLOCK_N=8, num_warps=2 for large N (lm_head); BLOCK_N=2, num_warps=4 for small N
    BLOCK_N = 8 if N >= 4096 else 2
    num_warps = 2 if N >= 4096 else 4
    BLOCK_M = min(1024, M)
    assert BLOCK_M % group_size == 0
    assert BLOCK_M % 2 == 0
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _gemv_int4_kernel[grid](
        x_flat, weight_int4, weight_scale, y,
        M, N,
        weight_int4.stride(0),
        num_groups,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )

    if len(orig_shape) > 1:
        return y.view(1, N)
    return y
