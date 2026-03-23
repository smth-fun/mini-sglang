"""Persistent forward kernel wrapper using CUDA cooperative groups.

Replaces the entire CUDA graph for bs=1 decode with a single cooperative kernel.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import load

if TYPE_CHECKING:
    from minisgl.attention.fi import FlashInferBackend
    from minisgl.models.qwen3 import Qwen3ForCausalLM

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module
    src = os.path.join(os.path.dirname(__file__), "csrc", "persistent_forward_v2.cu")
    _module = load(
        name="persistent_forward_v2",
        sources=[src],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _module


class PersistentForward:
    """Manages the persistent forward kernel for bs=1 decode."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        attn_backend: FlashInferBackend,
        device: torch.device,
    ):
        self.device = device
        self.mod = _get_module()

        # Extract model weights
        layers = model.model.layers.op_list

        # Per-layer weight lists
        input_norm_w = []
        qkv_w = []
        qkv_scale = []
        q_norm_w = []
        k_norm_w = []
        o_w = []
        o_scale = []
        post_norm_w = []
        gate_up_w = []
        gate_up_scale = []
        down_w = []
        down_scale = []
        k_cache_list = []
        v_cache_list = []

        kvcache = attn_backend.kvcache
        storage_shape = kvcache._storage_shape

        for i, layer in enumerate(layers):
            input_norm_w.append(layer.input_layernorm.weight)
            qkv_w.append(layer.self_attn.qkv_proj.weight_int8)
            qkv_scale.append(layer.self_attn.qkv_proj.weight_scale)
            q_norm_w.append(layer.self_attn.q_norm.weight)
            k_norm_w.append(layer.self_attn.k_norm.weight)
            o_w.append(layer.self_attn.o_proj.weight_int8)
            o_scale.append(layer.self_attn.o_proj.weight_scale)
            post_norm_w.append(layer.post_attention_layernorm.weight)
            gate_up_w.append(layer.mlp.gate_up_proj.weight_int8)
            gate_up_scale.append(layer.mlp.gate_up_proj.weight_scale)
            down_w.append(layer.mlp.down_proj.weight_int8)
            down_scale.append(layer.mlp.down_proj.weight_scale)
            k_cache_list.append(kvcache.k_cache(i).view(storage_shape))
            v_cache_list.append(kvcache.v_cache(i).view(storage_shape))

        # Embedding and final layers
        embed_w = model.model.embed_tokens.weight
        final_norm_w = model.model.norm.weight
        lm_head = model.lm_head
        lm_head_w_int8 = lm_head.weight_int8
        lm_head_scale = lm_head.weight_scale

        # RoPE cache
        rotary = layers[0].self_attn.attn.rotary
        cos_sin_cache = rotary._cos_sin_cache  # (max_pos, head_dim) float32
        cos_sin_stride = cos_sin_cache.stride(0)

        # Attention params
        config = attn_backend.config
        head_dim = config.head_dim
        num_kv = attn_backend.kv_head_local
        cache_stride_token = num_kv * head_dim
        sm_scale = 1.0 / (head_dim ** 0.5)

        # Reuse split-K buffers from attn_backend
        page_indices = attn_backend._splitk_page_indices
        seq_len_tensor = attn_backend._splitk_seq_len

        # Allocate scratch buffers
        hidden = config.hidden_size
        inter = config.intermediate_size
        num_qo = attn_backend.qo_head_local
        vocab_size = config.vocab_size

        self.x_buf = torch.empty(hidden, dtype=torch.bfloat16, device=device)
        self.residual_buf_a = torch.empty(hidden, dtype=torch.bfloat16, device=device)
        self.residual_buf_b = torch.empty(hidden, dtype=torch.bfloat16, device=device)
        qkv_dim = (num_qo + 2 * num_kv) * head_dim  # 4096
        self.qkv_buf = torch.empty(qkv_dim, dtype=torch.bfloat16, device=device)
        self.attn_out_buf = torch.empty(num_qo * head_dim, dtype=torch.bfloat16, device=device)
        self.mlp_hidden_buf = torch.empty(inter, dtype=torch.bfloat16, device=device)
        self.partial_out_buf = torch.empty(num_qo * 16 * head_dim, dtype=torch.float32, device=device)
        self.partial_lse_buf = torch.empty(num_qo * 16, dtype=torch.float32, device=device)
        self.logits_buf = torch.empty(vocab_size, dtype=torch.float32, device=device)

        # Dynamic input buffers (will be updated each step)
        self.input_ids_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.positions_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.out_loc_buf = torch.zeros(1, dtype=torch.int32, device=device)

        eps = layers[0].input_layernorm.eps

        # Save references for replay
        self.page_indices = page_indices
        self.seq_len_tensor = seq_len_tensor
        self.attn_backend = attn_backend

        # Initialize the CUDA extension
        info = self.mod.init_persistent_forward(
            embed_w,
            input_norm_w, qkv_w, qkv_scale,
            q_norm_w, k_norm_w,
            o_w, o_scale,
            post_norm_w,
            gate_up_w, gate_up_scale,
            down_w, down_scale,
            k_cache_list, v_cache_list,
            final_norm_w, lm_head_w_int8, lm_head_scale,
            eps,
            cos_sin_cache, cos_sin_stride,
            page_indices, seq_len_tensor,
            cache_stride_token, sm_scale,
            self.input_ids_buf, self.positions_buf, self.out_loc_buf,
            self.x_buf, self.residual_buf_a, self.residual_buf_b,
            self.qkv_buf, self.attn_out_buf, self.mlp_hidden_buf,
            self.partial_out_buf, self.partial_lse_buf,
            self.logits_buf,
            vocab_size,
        )
        self.num_blocks = info[0].item()
        print(f"[PersistentForward] Initialized with {self.num_blocks} blocks")

    def run(self, batch) -> torch.Tensor:
        """Run persistent forward for bs=1 decode.

        Updates dynamic inputs and launches the cooperative kernel.
        Returns logits tensor (1, vocab_size) float32.
        """
        # Update dynamic inputs
        self.input_ids_buf[0] = batch.input_ids[0]
        self.positions_buf[0] = batch.positions[0]
        self.out_loc_buf[0] = batch.out_loc[0]

        # Update attention metadata (page indices and seq_len)
        metadata = batch.attn_metadata
        seq_len = metadata.seq_lens_cpu[0].item()
        self.page_indices[:seq_len].copy_(metadata.indices[:seq_len], non_blocking=True)
        self.seq_len_tensor[0] = seq_len

        # Launch kernel
        self.mod.launch_persistent_forward()

        return self.logits_buf.unsqueeze(0)  # (1, vocab_size)
