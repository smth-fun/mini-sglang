from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            if qkv.shape[0] == 1:
                backend = ctx.attn_backend
                kvcache = ctx.kv_cache
                k_cache = kvcache.k_cache(self.layer_id).view(kvcache._storage_shape)
                v_cache = kvcache.v_cache(self.layer_id).view(kvcache._storage_shape)

                # Fused QKNorm + RoPE + KV store + Split-K attention (1 kernel instead of 3)
                if getattr(backend, '_use_splitk', False):
                    from minisgl.kernel.fused_qknorm_attn import fused_qknorm_splitk_attention_forward
                    o = fused_qknorm_splitk_attention_forward(
                        qkv,
                        self.q_norm.weight, self.k_norm.weight,
                        self.rotary._cos_sin_cache,
                        ctx.batch.positions, ctx.batch.out_loc,
                        k_cache, v_cache,
                        backend._splitk_page_indices,
                        backend._splitk_seq_len,
                        self.num_qo_heads, self.num_kv_heads, self.head_dim,
                        backend._splitk_sm_scale,
                        self.q_norm.eps,
                        backend._splitk_partial_out,
                        backend._splitk_partial_lse,
                        backend._splitk_out,
                        backend._splitk_q_scratch,
                        backend._splitk_head_counter,
                    )
                    return o.view(-1, self.qo_attn_dim)

                # Fallback: separate QKNorm + RoPE + KV store, then attention
                from minisgl.kernel.fused_qk_rope import fused_qk_norm_rope_store
                fused_qk_norm_rope_store(
                    qkv.view(-1),
                    self.q_norm.weight, self.k_norm.weight,
                    self.rotary._cos_sin_cache,
                    ctx.batch.positions, ctx.batch.out_loc,
                    k_cache, v_cache,
                    self.q_norm.eps,
                    self.num_qo_heads, self.num_kv_heads, self.head_dim,
                )
                q = q.view(-1, self.num_qo_heads, self.head_dim)
                o = backend.forward(q, k, v, self.layer_id, ctx.batch, skip_store=True)
                return o.view(-1, self.qo_attn_dim)
            else:
                self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
                if self.k_norm is not None:
                    self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        elif self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
