"""Fused QKNorm + RoPE + KV Store kernel for decode (bs=1).

Multi-block version: one block per head (Q or K).
- Q head blocks: RMSNorm + RoPE in-place
- K head blocks: RMSNorm + RoPE in-place + store K to cache + store V to cache

Eliminates 2 separate kernel launches (RoPE + store_kv) per layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    pass


@triton.jit
def _fused_qk_norm_rope_store_kernel(
    # QKV pointers (contiguous: [Q_heads | K_heads | V_heads])
    qkv_ptr,
    # Norm weights
    q_nw_ptr, k_nw_ptr,
    # RoPE cos/sin cache
    cos_sin_ptr,
    # Position and out_loc
    pos_ptr, out_loc_ptr,
    # KV cache pointers (for this layer, shaped as (num_pages, kv_heads, head_dim))
    k_cache_ptr, v_cache_ptr,
    # Strides
    cos_sin_stride,  # stride of cos_sin_cache along position dim
    cache_stride_token,  # stride per token in KV cache (kv_heads * head_dim)
    # Constants
    eps: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    QO_DIM: tl.constexpr,  # NUM_QO_HEADS * HEAD_DIM (offset to K in QKV)
    KV_DIM: tl.constexpr,  # NUM_KV_HEADS * HEAD_DIM (offset from K to V in QKV)
):
    pid = tl.program_id(0)
    half_ids = tl.arange(0, HALF_DIM)

    # Load position and cache location (shared across all blocks)
    position = tl.load(pos_ptr)
    out_loc = tl.load(out_loc_ptr)

    # Load RoPE cos/sin for this position
    cs_base = cos_sin_ptr + position * cos_sin_stride
    cos_vals = tl.load(cs_base + half_ids).to(tl.float32)
    sin_vals = tl.load(cs_base + HALF_DIM + half_ids).to(tl.float32)

    is_q = pid < NUM_QO_HEADS

    if is_q:
        # --- Q head: RMSNorm + RoPE in-place ---
        head_offset = pid * HEAD_DIM
        base_ptr = qkv_ptr + head_offset

        # Load norm weights
        nw_lo = tl.load(q_nw_ptr + half_ids).to(tl.float32)
        nw_hi = tl.load(q_nw_ptr + HALF_DIM + half_ids).to(tl.float32)

        # Load Q head data
        lo = tl.load(base_ptr + half_ids).to(tl.float32)
        hi = tl.load(base_ptr + HALF_DIM + half_ids).to(tl.float32)

        # RMSNorm
        ss = tl.sum(lo * lo) + tl.sum(hi * hi)
        rrms = 1.0 / tl.sqrt(ss / HEAD_DIM + eps)
        lo = lo * rrms * nw_lo
        hi = hi * rrms * nw_hi

        # RoPE
        new_lo = lo * cos_vals - hi * sin_vals
        new_hi = hi * cos_vals + lo * sin_vals

        # Store back in-place
        tl.store(base_ptr + half_ids, new_lo.to(base_ptr.dtype.element_ty))
        tl.store(base_ptr + HALF_DIM + half_ids, new_hi.to(base_ptr.dtype.element_ty))
    else:
        # --- K head: RMSNorm + RoPE in-place + store K,V to cache ---
        kv_head_idx = pid - NUM_QO_HEADS
        k_head_offset = QO_DIM + kv_head_idx * HEAD_DIM
        k_base_ptr = qkv_ptr + k_head_offset

        # Load norm weights
        nw_lo = tl.load(k_nw_ptr + half_ids).to(tl.float32)
        nw_hi = tl.load(k_nw_ptr + HALF_DIM + half_ids).to(tl.float32)

        # Load K head data
        lo = tl.load(k_base_ptr + half_ids).to(tl.float32)
        hi = tl.load(k_base_ptr + HALF_DIM + half_ids).to(tl.float32)

        # RMSNorm
        ss = tl.sum(lo * lo) + tl.sum(hi * hi)
        rrms = 1.0 / tl.sqrt(ss / HEAD_DIM + eps)
        lo = lo * rrms * nw_lo
        hi = hi * rrms * nw_hi

        # RoPE
        new_lo = lo * cos_vals - hi * sin_vals
        new_hi = hi * cos_vals + lo * sin_vals

        # Store K back in-place (for attention to read)
        tl.store(k_base_ptr + half_ids, new_lo.to(k_base_ptr.dtype.element_ty))
        tl.store(k_base_ptr + HALF_DIM + half_ids, new_hi.to(k_base_ptr.dtype.element_ty))

        # Store K to cache
        cache_base = k_cache_ptr + out_loc * cache_stride_token + kv_head_idx * HEAD_DIM
        tl.store(cache_base + half_ids, new_lo.to(k_cache_ptr.dtype.element_ty))
        tl.store(cache_base + HALF_DIM + half_ids, new_hi.to(k_cache_ptr.dtype.element_ty))

        # Store V to cache (no norm/RoPE needed)
        v_head_offset = QO_DIM + KV_DIM + kv_head_idx * HEAD_DIM
        full_ids = tl.arange(0, HEAD_DIM)
        v_vals = tl.load(qkv_ptr + v_head_offset + full_ids)
        v_cache_base = v_cache_ptr + out_loc * cache_stride_token + kv_head_idx * HEAD_DIM
        tl.store(v_cache_base + full_ids, v_vals)


def fused_qk_norm_rope_store(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Fused QKNorm + RoPE + KV store for bs=1 decode.

    Modifies Q and K in-place (applies norm + RoPE).
    Stores K and V to KV cache.

    Args:
        qkv: flat QKV tensor, contiguous [Q_heads | K_heads | V_heads]
        k_cache: (num_pages, kv_heads, head_dim) or equivalent flat view
        v_cache: same shape as k_cache
    """
    half_dim = head_dim // 2
    qo_dim = num_qo_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    cache_stride_token = k_cache.stride(0) if k_cache.dim() >= 2 else num_kv_heads * head_dim

    grid = (num_qo_heads + num_kv_heads,)
    _fused_qk_norm_rope_store_kernel[grid](
        qkv,
        q_norm_weight, k_norm_weight,
        cos_sin_cache,
        positions, out_loc,
        k_cache, v_cache,
        cos_sin_cache.stride(0),
        cache_stride_token,
        eps=eps,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        QO_DIM=qo_dim,
        KV_DIM=kv_dim,
        num_warps=1,
    )
