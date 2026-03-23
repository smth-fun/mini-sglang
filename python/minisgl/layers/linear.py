from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.kernel.triton_gemv import triton_gemv, _should_use_triton, _get_config
from minisgl.utils import div_even

from .base import BaseOP


def _fast_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, weight_int8: torch.Tensor | None = None, weight_scale: torch.Tensor | None = None, weight_int4: torch.Tensor | None = None, weight_scale_int4: torch.Tensor | None = None) -> torch.Tensor:
    """Use Triton GEMV for bs=1 decode where it's faster, cuBLAS otherwise."""
    if x.shape[0] == 1 and bias is None:
        # Prefer INT4 over INT8 for maximum bandwidth reduction
        if weight_int4 is not None and _should_use_triton(weight.shape[1], weight.shape[0]):
            from minisgl.kernel.triton_gemv_int4 import triton_gemv_int4
            return triton_gemv_int4(x, weight_int4, weight_scale_int4, group_size=128)
        if weight_int8 is not None and _should_use_triton(weight.shape[1], weight.shape[0]):
            return triton_gemv(x, weight_int8, weight_scale=weight_scale)
        if _should_use_triton(weight.shape[1], weight.shape[0]):
            return triton_gemv(x, weight)
    return F.linear(x, weight, bias)


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fast_linear(x, self.weight, self.bias,
                            getattr(self, 'weight_int8', None),
                            getattr(self, 'weight_scale', None),
                            getattr(self, 'weight_int4', None),
                            getattr(self, 'weight_scale_int4', None))


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _fast_linear(x, self.weight, self.bias,
                         getattr(self, 'weight_int8', None),
                         getattr(self, 'weight_scale', None),
                         getattr(self, 'weight_int4', None),
                         getattr(self, 'weight_scale_int4', None))
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _fast_linear(x, self.weight, self.bias,
                         getattr(self, 'weight_int8', None),
                         getattr(self, 'weight_scale', None),
                         getattr(self, 'weight_int4', None),
                         getattr(self, 'weight_scale_int4', None))
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
