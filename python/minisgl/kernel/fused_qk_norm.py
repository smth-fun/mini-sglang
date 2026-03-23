"""Combined Q + K RMSNorm in a single kernel launch.

Replaces two separate FlashInfer QKRMSNorm calls with one Triton kernel.
Each block handles one head (Q or K), normalizing head_dim elements in-place.

Layout: Q and K are contiguous in the QKV buffer:
  QKV = [Q_head0, Q_head1, ..., Q_head15, K_head0, ..., K_head7, V_head0, ...]
So pid * HEAD_DIM directly indexes into the correct head.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _combined_qk_rmsnorm_kernel(
    qkv_ptr, q_nw_ptr, k_nw_ptr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps: tl.constexpr,
):
    """Normalize Q and K heads in one kernel. V is untouched.

    qkv_ptr points to start of Q (= start of QKV buffer).
    Heads 0..num_q_heads-1 are Q, heads num_q_heads..num_q_heads+num_k_heads-1 are K.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, HEAD_DIM)

    # All heads are contiguous: head i starts at offset i * HEAD_DIM
    head_offset = pid * HEAD_DIM
    vals = tl.load(qkv_ptr + head_offset + cols).to(tl.float32)

    # RMS norm
    sq_sum = tl.sum(vals * vals)
    rms_inv = tl.rsqrt(sq_sum / HEAD_DIM + eps)

    # Load both norm weights (cheap, only HEAD_DIM=64 elements each)
    nw_q = tl.load(q_nw_ptr + cols).to(tl.float32)
    nw_k = tl.load(k_nw_ptr + cols).to(tl.float32)

    # Select norm weight based on whether this is a Q or K head
    is_q = pid < num_q_heads
    nw = tl.where(is_q, nw_q, nw_k)

    normed = vals * rms_inv * nw
    tl.store(qkv_ptr + head_offset + cols, normed.to(qkv_ptr.dtype.element_ty))


def combined_qk_rmsnorm_inplace(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Combined in-place Q and K RMSNorm on the QKV buffer.

    Args:
        qkv: flat QKV tensor (total_dim,) where Q, K, V are contiguous segments
        q_norm_weight: (head_dim,)
        k_norm_weight: (head_dim,)
    """
    grid = (num_q_heads + num_k_heads,)
    _combined_qk_rmsnorm_kernel[grid](
        qkv, q_norm_weight, k_norm_weight,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        HEAD_DIM=head_dim,
        eps=eps,
        num_warps=1,
    )
