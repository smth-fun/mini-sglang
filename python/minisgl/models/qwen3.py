from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id
        self._hidden_size = config.hidden_size
        self._use_fused_decode = (config.hidden_act == "silu")
        # Lazy-init residual buffers for fused decode path
        self._res_buf_a: torch.Tensor | None = None
        self._res_buf_b: torch.Tensor | None = None

    def _init_res_bufs(self, x: torch.Tensor) -> None:
        self._res_buf_a = torch.empty(1, self._hidden_size, device=x.device, dtype=x.dtype)
        self._res_buf_b = torch.empty(1, self._hidden_size, device=x.device, dtype=x.dtype)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fused decode path: eliminate separate FusedAddRMSNorm kernels
        if self._use_fused_decode and x.shape[0] == 1:
            return self._forward_decode(x, residual)
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual

    def _forward_decode(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from minisgl.kernel.fused_norm_gemv import (
            fused_add_rmsnorm_gemv,
            fused_add_rmsnorm_gemv_silu,
            fused_rmsnorm_gemv,
        )

        if self._res_buf_a is None:
            self._init_res_bufs(x)

        # Get INT8 quantized weights if available
        qkv_proj = self.self_attn.qkv_proj
        qkv_w = getattr(qkv_proj, 'weight_int8', qkv_proj.weight)
        qkv_s = getattr(qkv_proj, 'weight_scale', None)

        # --- Input layernorm + QKV GEMV (fused) ---
        if residual is not None:
            qkv = fused_add_rmsnorm_gemv(
                x, residual,
                self.input_layernorm.weight,
                qkv_w,
                self.input_layernorm.eps,
                self._res_buf_a,
                weight_scale=qkv_s,
            )
            residual = self._res_buf_a
        else:
            # First layer: no residual to add
            qkv = fused_rmsnorm_gemv(
                x,
                self.input_layernorm.weight,
                qkv_w,
                self.input_layernorm.eps,
                weight_scale=qkv_s,
            )
            residual = x  # embedding becomes the residual

        # --- Attention (QKNorm + RoPE + decode + O proj) ---
        x = self.self_attn.forward_post_qkv(qkv)

        # --- Post-attention layernorm + gate_up GEMV + silu (fused) ---
        gate_up_proj = self.mlp.gate_up_proj
        gate_up_w = getattr(gate_up_proj, 'weight_int8', gate_up_proj.weight)
        gate_up_s = getattr(gate_up_proj, 'weight_scale', None)

        y = fused_add_rmsnorm_gemv_silu(
            x, residual,
            self.post_attention_layernorm.weight,
            gate_up_w,
            self.post_attention_layernorm.eps,
            self._res_buf_b,
            weight_scale=gate_up_s,
        )

        # --- Down projection (INT4 handled by _fast_linear in Linear.forward) ---
        x = self.mlp.down_proj.forward(y)
        return x, self._res_buf_b


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._lm_head_res_buf: torch.Tensor | None = None
        super().__init__()

    def forward(self) -> torch.Tensor:
        input_ids = get_global_ctx().batch.input_ids
        x = self.model.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.model.layers.op_list:
            x, residual = layer.forward(x, residual)

        # Fuse final norm + INT4 lm_head GEMV for bs=1 decode (saves 1 kernel)
        w_int4 = getattr(self.lm_head, 'weight_int4', None)
        if x.shape[0] == 1 and w_int4 is not None and residual is not None:
            from minisgl.kernel.fused_norm_gemv_int4 import fused_add_rmsnorm_gemv_int4
            if self._lm_head_res_buf is None:
                self._lm_head_res_buf = torch.empty_like(residual)
            return fused_add_rmsnorm_gemv_int4(
                x, residual, self.model.norm.weight,
                w_int4, self.lm_head.weight_scale_int4,
                self.model.norm.eps,
                residual_out=self._lm_head_res_buf,
            )

        output = self.model.norm.forward(x, residual)[0]
        return self.lm_head.forward(output)


__all__ = ["Qwen3ForCausalLM"]
