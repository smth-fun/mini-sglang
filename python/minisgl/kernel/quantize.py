"""INT8 per-row and INT4 per-group weight quantization for decode GEMV acceleration."""

from __future__ import annotations

from typing import Tuple

import torch


def quantize_weight_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quantization.

    Args:
        weight: bf16/fp16 weight tensor of shape (N, M)

    Returns:
        (weight_int8, scale) where:
        - weight_int8: int8 tensor of shape (N, M), contiguous
        - scale: float32 tensor of shape (N,) - per-row scale factors
        Dequantization: weight_approx = weight_int8.float() * scale.unsqueeze(1)
    """
    w_f32 = weight.float()
    abs_max = w_f32.abs().amax(dim=1)  # (N,)
    scale = abs_max / 127.0
    scale = scale.clamp(min=1e-10)  # avoid division by zero
    weight_int8 = (w_f32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return weight_int8.contiguous(), scale.contiguous()


def quantize_weight_int4(
    weight: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric INT4 quantization, packed 2 values per uint8 byte.

    Args:
        weight: bf16/fp16 weight tensor of shape (N, M)
        group_size: number of elements per quantization group

    Returns:
        (weight_int4_packed, group_scale) where:
        - weight_int4_packed: uint8 tensor of shape (N, M//2), contiguous
          Packing: byte = ((val0 + 8) & 0xF) | (((val1 + 8) & 0xF) << 4)
          where val0, val1 are signed int4 in [-8, 7]
          val0 = even-index element, val1 = odd-index element
        - group_scale: float16 tensor of shape (N, M // group_size), contiguous
    """
    N, M = weight.shape
    assert M % group_size == 0, f"M={M} must be divisible by group_size={group_size}"
    assert M % 2 == 0, f"M={M} must be even for int4 packing"

    w_f32 = weight.float()
    num_groups = M // group_size

    # Reshape to (N, num_groups, group_size) for per-group quantization
    w_grouped = w_f32.view(N, num_groups, group_size)
    abs_max = w_grouped.abs().amax(dim=2)  # (N, num_groups)
    scale = abs_max / 7.0  # 4-bit signed: [-8, 7], use 7 for symmetric
    scale = scale.clamp(min=1e-10)

    # Quantize each group
    w_quant = (w_grouped / scale.unsqueeze(2)).round().clamp(-8, 7)  # (N, num_groups, group_size)
    w_quant = w_quant.view(N, M).to(torch.int8)  # (N, M)

    # Pack pairs of int4 into uint8: even-index in low nibble, odd-index in high nibble
    # Shift to unsigned [0, 15] range first
    w_unsigned = (w_quant + 8).to(torch.uint8)  # [0, 15]
    w_even = w_unsigned[:, 0::2]  # (N, M//2) - even indices
    w_odd = w_unsigned[:, 1::2]   # (N, M//2) - odd indices
    w_packed = w_even | (w_odd << 4)  # (N, M//2) uint8

    return w_packed.contiguous(), scale.to(torch.float16).contiguous()


def quantize_model_weights(model, skip_decoder: bool = False, use_int4: bool = False, int4_group_size: int = 128) -> None:
    """Quantize GEMV weights to INT8, optionally also to INT4.

    Adds .weight_int8 and .weight_scale attributes to each Linear layer.
    If use_int4=True, also adds .weight_int4 and .weight_scale_int4 attributes.
    Keeps original bf16 .weight for prefill path (cuBLAS) and embedding lookup.
    """
    if not skip_decoder:
        for layer in model.model.layers.op_list:
            _quantize_linear(layer.self_attn.qkv_proj, use_int4=use_int4, group_size=int4_group_size)
            _quantize_linear(layer.self_attn.o_proj, use_int4=use_int4, group_size=int4_group_size)
            _quantize_linear(layer.mlp.gate_up_proj, use_int4=use_int4, group_size=int4_group_size)
            _quantize_linear(layer.mlp.down_proj, use_int4=use_int4, group_size=int4_group_size)

    # Quantize lm_head weight (may be tied to embedding)
    lm_head = model.lm_head
    lm_module = lm_head.tied_embedding or lm_head
    lm_head.weight_int8, lm_head.weight_scale = quantize_weight_int8(lm_module.weight)
    if use_int4:
        lm_head.weight_int4, lm_head.weight_scale_int4 = quantize_weight_int4(
            lm_module.weight, group_size=int4_group_size
        )


def _quantize_linear(linear, use_int4: bool = False, group_size: int = 128) -> None:
    linear.weight_int8, linear.weight_scale = quantize_weight_int8(linear.weight)
    if use_int4:
        linear.weight_int4, linear.weight_scale_int4 = quantize_weight_int4(
            linear.weight, group_size=group_size
        )
