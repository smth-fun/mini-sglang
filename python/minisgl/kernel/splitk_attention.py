"""Split-K Triton decode attention kernel.

Parallelizes across both heads AND sequence chunks for better SM utilization.
For bs=1 decode with Qwen3-0.6B: grid=(16, 32) = 512 blocks on 188 SMs.
Replaces FlashInfer decode attention, saving ~1-2μs per layer (14-26% faster).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_attn_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    partial_out_ptr, partial_lse_ptr,
    page_indices_ptr, seq_len_ptr, sm_scale, cache_stride_token,
    GQA_RATIO: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, NUM_CHUNKS: tl.constexpr,
):
    """Phase 1: Each block computes partial attention for one (head, chunk) pair."""
    qo_head = tl.program_id(0)
    chunk_id = tl.program_id(1)
    kv_head = qo_head // GQA_RATIO

    seq_len = tl.load(seq_len_ptr)

    dim_ids = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + qo_head * HEAD_DIM + dim_ids).to(tl.float32)

    seq_start = chunk_id * BLOCK_SEQ
    seq_ids = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_ids < seq_len

    pages = tl.load(page_indices_ptr + seq_ids, mask=seq_mask, other=0)
    k_base = pages.to(tl.int64) * cache_stride_token + kv_head * HEAD_DIM

    k_offsets = k_base[:, None] + dim_ids[None, :]
    k_vals = tl.load(k_cache_ptr + k_offsets, mask=seq_mask[:, None], other=0.0).to(tl.float32)

    scores = tl.sum(k_vals * q[None, :], axis=1) * sm_scale
    scores = tl.where(seq_mask, scores, -1e30)

    m = tl.max(scores)
    exp_scores = tl.exp(scores - m)
    l = tl.sum(exp_scores)

    v_vals = tl.load(v_cache_ptr + k_offsets, mask=seq_mask[:, None], other=0.0).to(tl.float32)
    acc = tl.sum(exp_scores[:, None] * v_vals, axis=0)

    # Normalize within chunk (critical for correct reduction)
    safe_l = tl.where(l > 0, l, 1.0)
    acc = acc / safe_l

    out_offset = qo_head * NUM_CHUNKS * HEAD_DIM + chunk_id * HEAD_DIM
    tl.store(partial_out_ptr + out_offset + dim_ids, acc.to(partial_out_ptr.dtype.element_ty))

    lse_offset = qo_head * NUM_CHUNKS + chunk_id
    lse = tl.where(l > 0, m + tl.log(l), -1e30)
    tl.store(partial_lse_ptr + lse_offset, lse)


@triton.jit
def _reduce_attn_kernel(
    partial_out_ptr, partial_lse_ptr, out_ptr,
    HEAD_DIM: tl.constexpr, NUM_CHUNKS: tl.constexpr,
):
    """Phase 2: Reduce partial attention across chunks for each head."""
    qo_head = tl.program_id(0)
    dim_ids = tl.arange(0, HEAD_DIM)
    chunk_ids = tl.arange(0, NUM_CHUNKS)

    lse_base = qo_head * NUM_CHUNKS
    lse_vals = tl.load(partial_lse_ptr + lse_base + chunk_ids)
    m_global = tl.max(lse_vals)

    total_weight = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for c in tl.static_range(0, NUM_CHUNKS):
        out_offset = qo_head * NUM_CHUNKS * HEAD_DIM + c * HEAD_DIM
        partial = tl.load(partial_out_ptr + out_offset + dim_ids).to(tl.float32)
        corr = tl.load(partial_lse_ptr + lse_base + c)
        w = tl.exp(corr - m_global)
        acc += partial * w
        total_weight += w

    acc = acc / total_weight
    tl.store(out_ptr + qo_head * HEAD_DIM + dim_ids, acc.to(out_ptr.dtype.element_ty))


# Pre-computed constants for Qwen3-0.6B
_BLOCK_SEQ = 128
_NUM_CHUNKS_PAD = 32  # covers up to 4096 tokens (128 * 32)


def splitk_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sm_scale: float,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run split-K attention for bs=1 decode.

    Args:
        q: (1, num_qo_heads, head_dim) query
        k_cache: (total_tokens, num_kv_heads, head_dim) - storage shape view
        v_cache: (total_tokens, num_kv_heads, head_dim) - storage shape view
        page_indices: (max_seq_len,) int32 page indices
        seq_len_tensor: (1,) int32 GPU tensor with current sequence length
        num_qo_heads, num_kv_heads, head_dim: model dimensions
        sm_scale: 1/sqrt(head_dim)
        partial_out: (num_qo_heads, NUM_CHUNKS_PAD, head_dim) pre-allocated float32
        partial_lse: (num_qo_heads, NUM_CHUNKS_PAD) pre-allocated float32
        out: (1, num_qo_heads, head_dim) pre-allocated output buffer

    Returns:
        out tensor (1, num_qo_heads, head_dim)
    """
    gqa_ratio = num_qo_heads // num_kv_heads
    cache_stride_token = num_kv_heads * head_dim

    _splitk_attn_kernel[(num_qo_heads, _NUM_CHUNKS_PAD)](
        q.view(-1), k_cache, v_cache,
        partial_out, partial_lse,
        page_indices, seq_len_tensor, sm_scale, cache_stride_token,
        GQA_RATIO=gqa_ratio, HEAD_DIM=head_dim,
        BLOCK_SEQ=_BLOCK_SEQ, NUM_CHUNKS=_NUM_CHUNKS_PAD,
        num_warps=4,
    )
    _reduce_attn_kernel[(num_qo_heads,)](
        partial_out, partial_lse, out.view(-1),
        HEAD_DIM=head_dim, NUM_CHUNKS=_NUM_CHUNKS_PAD,
        num_warps=1,
    )
    return out
