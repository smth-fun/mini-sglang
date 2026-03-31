"""Fused QKNorm + RoPE + KV Store + Split-K Attention + Reduce kernel for decode (bs=1).

Combines fused_qk_norm_rope_store + _splitk_attn_kernel + _reduce_attn_kernel into
a single kernel launch. Each attention block redundantly computes Q norm + RoPE.
Blocks in the chunk containing the current position also store K,V to cache.
Last block per head does the reduction inline (atomic counter with threadfence).

Eliminates 2 kernel launches per layer = 56 per step.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Pre-computed constants for Qwen3-0.6B
_BLOCK_SEQ = 128
_NUM_CHUNKS_PAD = 32  # covers up to 4096 tokens (128 * 32)


@triton.jit
def _fused_qknorm_splitk_reduce_kernel(
    # QKV buffer (flat: [Q_heads | K_heads | V_heads])
    qkv_ptr,
    # Norm weights
    q_nw_ptr, k_nw_ptr,
    # RoPE cos/sin cache
    cos_sin_ptr,
    # Position and out_loc
    pos_ptr, out_loc_ptr,
    # KV cache
    k_cache_ptr, v_cache_ptr,
    # Split-K attention buffers
    partial_out_ptr, partial_lse_ptr,
    # Q scratch buffer (num_qo_heads, HEAD_DIM) float32
    q_scratch_ptr,
    # Output buffer
    out_ptr,
    # Atomic counter buffer (num_qo_heads,) int32
    head_counter_ptr,
    # Page indices and seq_len
    page_indices_ptr, seq_len_ptr,
    # Scalars
    sm_scale,
    cos_sin_stride,
    cache_stride_token,
    # Constants
    eps: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    QO_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Fused QK-Norm + RoPE + KV-Store + Split-K Attention + Last-Block Reduce."""
    qo_head = tl.program_id(0)
    chunk_id = tl.program_id(1)
    kv_head = qo_head // GQA_RATIO

    half_ids = tl.arange(0, HALF_DIM)
    dim_ids = tl.arange(0, HEAD_DIM)

    # --- Step 1: Q norm + RoPE → write to scratch buffer ---
    q_base = qkv_ptr + qo_head * HEAD_DIM
    q_lo = tl.load(q_base + half_ids).to(tl.float32)
    q_hi = tl.load(q_base + HALF_DIM + half_ids).to(tl.float32)

    q_nw_lo = tl.load(q_nw_ptr + half_ids).to(tl.float32)
    q_nw_hi = tl.load(q_nw_ptr + HALF_DIM + half_ids).to(tl.float32)

    # RMSNorm Q
    q_ss = tl.sum(q_lo * q_lo) + tl.sum(q_hi * q_hi)
    q_rrms = 1.0 / tl.sqrt(q_ss / HEAD_DIM + eps)
    q_lo = q_lo * q_rrms * q_nw_lo
    q_hi = q_hi * q_rrms * q_nw_hi

    # Load RoPE cos/sin
    position = tl.load(pos_ptr)
    cs_base = cos_sin_ptr + position * cos_sin_stride
    cos_vals = tl.load(cs_base + half_ids).to(tl.float32)
    sin_vals = tl.load(cs_base + HALF_DIM + half_ids).to(tl.float32)

    # Apply RoPE to Q
    q_roped_lo = q_lo * cos_vals - q_hi * sin_vals
    q_roped_hi = q_hi * cos_vals + q_lo * sin_vals

    # Write normalized+RoPE'd Q to scratch
    q_scratch_base = q_scratch_ptr + qo_head * HEAD_DIM
    tl.store(q_scratch_base + half_ids, q_roped_lo)
    tl.store(q_scratch_base + HALF_DIM + half_ids, q_roped_hi)

    # --- Step 2: K norm + RoPE + Store K,V (only for blocks in current chunk) ---
    seq_len = tl.load(seq_len_ptr)
    current_chunk = (seq_len - 1) // BLOCK_SEQ

    if chunk_id == current_chunk:
        k_head_offset = QO_DIM + kv_head * HEAD_DIM
        k_base_ptr = qkv_ptr + k_head_offset
        k_lo = tl.load(k_base_ptr + half_ids).to(tl.float32)
        k_hi = tl.load(k_base_ptr + HALF_DIM + half_ids).to(tl.float32)

        k_nw_lo = tl.load(k_nw_ptr + half_ids).to(tl.float32)
        k_nw_hi = tl.load(k_nw_ptr + HALF_DIM + half_ids).to(tl.float32)

        k_ss = tl.sum(k_lo * k_lo) + tl.sum(k_hi * k_hi)
        k_rrms = 1.0 / tl.sqrt(k_ss / HEAD_DIM + eps)
        k_lo = k_lo * k_rrms * k_nw_lo
        k_hi = k_hi * k_rrms * k_nw_hi

        k_new_lo = k_lo * cos_vals - k_hi * sin_vals
        k_new_hi = k_hi * cos_vals + k_lo * sin_vals

        out_loc = tl.load(out_loc_ptr)
        cache_base = k_cache_ptr + out_loc * cache_stride_token + kv_head * HEAD_DIM
        tl.store(cache_base + half_ids, k_new_lo.to(k_cache_ptr.dtype.element_ty))
        tl.store(cache_base + HALF_DIM + half_ids, k_new_hi.to(k_cache_ptr.dtype.element_ty))

        v_head_offset = QO_DIM + KV_DIM + kv_head * HEAD_DIM
        v_vals = tl.load(qkv_ptr + v_head_offset + dim_ids)
        v_cache_base = v_cache_ptr + out_loc * cache_stride_token + kv_head * HEAD_DIM
        tl.store(v_cache_base + dim_ids, v_vals)

    # --- Step 3: Split-K Attention ---
    q = tl.load(q_scratch_base + dim_ids).to(tl.float32)

    seq_start = chunk_id * BLOCK_SEQ
    seq_ids = seq_start + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_ids < seq_len

    pages = tl.load(page_indices_ptr + seq_ids, mask=seq_mask, other=0)
    k_base_off = pages.to(tl.int64) * cache_stride_token + kv_head * HEAD_DIM

    k_offsets = k_base_off[:, None] + dim_ids[None, :]
    k_vals = tl.load(k_cache_ptr + k_offsets, mask=seq_mask[:, None], other=0.0).to(tl.float32)

    scores = tl.sum(k_vals * q[None, :], axis=1) * sm_scale
    scores = tl.where(seq_mask, scores, -1e30)

    m = tl.max(scores)
    exp_scores = tl.exp(scores - m)
    l = tl.sum(exp_scores)

    v_vals = tl.load(v_cache_ptr + k_offsets, mask=seq_mask[:, None], other=0.0).to(tl.float32)
    acc = tl.sum(exp_scores[:, None] * v_vals, axis=0)

    safe_l = tl.where(l > 0, l, 1.0)
    acc = acc / safe_l

    out_offset = qo_head * NUM_CHUNKS * HEAD_DIM + chunk_id * HEAD_DIM
    tl.store(partial_out_ptr + out_offset + dim_ids, acc.to(partial_out_ptr.dtype.element_ty))

    lse_offset = qo_head * NUM_CHUNKS + chunk_id
    lse = tl.where(l > 0, m + tl.log(l), -1e30)
    tl.store(partial_lse_ptr + lse_offset, lse)

    # --- Step 4: Last-block reduce ---
    # atomic_add provides ordering for the counter; partial_out/lse stores
    # commit through L2 before the counter is observed by the reduce block
    old_count = tl.atomic_add(head_counter_ptr + qo_head, 1)

    if (old_count + 1) % NUM_CHUNKS == 0:
        # This is the last block for this head - do reduction
        chunk_ids = tl.arange(0, NUM_CHUNKS)
        lse_base = qo_head * NUM_CHUNKS
        lse_vals = tl.load(partial_lse_ptr + lse_base + chunk_ids)
        m_global = tl.max(lse_vals)

        total_weight = 0.0
        final_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for c in tl.static_range(0, NUM_CHUNKS):
            p_offset = qo_head * NUM_CHUNKS * HEAD_DIM + c * HEAD_DIM
            partial = tl.load(partial_out_ptr + p_offset + dim_ids).to(tl.float32)
            corr = tl.load(partial_lse_ptr + lse_base + c)
            w = tl.exp(corr - m_global)
            final_acc += partial * w
            total_weight += w

        final_acc = final_acc / total_weight
        tl.store(out_ptr + qo_head * HEAD_DIM + dim_ids, final_acc.to(out_ptr.dtype.element_ty))


def fused_qknorm_splitk_attention_forward(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sm_scale: float,
    eps: float,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    q_scratch: torch.Tensor,
    head_counter: torch.Tensor,
) -> torch.Tensor:
    """Fused QKNorm + RoPE + KV Store + Split-K Attention + Reduce for bs=1 decode.

    Replaces: fused_qk_norm_rope_store + splitk_attention_forward (3 kernels → 1).
    """
    gqa_ratio = num_qo_heads // num_kv_heads
    half_dim = head_dim // 2
    qo_dim = num_qo_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    cache_stride_token = num_kv_heads * head_dim

    _fused_qknorm_splitk_reduce_kernel[(num_qo_heads, _NUM_CHUNKS_PAD)](
        qkv.view(-1),
        q_norm_weight, k_norm_weight,
        cos_sin_cache,
        positions, out_loc,
        k_cache, v_cache,
        partial_out, partial_lse,
        q_scratch,
        out.view(-1),
        head_counter,
        page_indices, seq_len_tensor,
        sm_scale,
        cos_sin_cache.stride(0),
        cache_stride_token,
        eps=eps,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_RATIO=gqa_ratio,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        QO_DIM=qo_dim,
        KV_DIM=kv_dim,
        BLOCK_SEQ=_BLOCK_SEQ,
        NUM_CHUNKS=_NUM_CHUNKS_PAD,
        num_warps=4,
    )

    return out
