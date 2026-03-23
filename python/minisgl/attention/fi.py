from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

        # Pre-allocated tensors for fast decode metadata path
        self._decode_cu_seqlens_k_cpu: torch.Tensor | None = None
        self._decode_seq_lens_cpu: torch.Tensor | None = None
        self._decode_cu_seqlens_q_cpu: torch.Tensor | None = None
        self._decode_cu_seqlens_q_gpu: torch.Tensor | None = None
        self._decode_last_page_cpu: torch.Tensor | None = None

        # Split-K attention buffers (allocated during graph capture init)
        self._splitk_partial_out: torch.Tensor | None = None
        self._splitk_partial_lse: torch.Tensor | None = None
        self._splitk_out: torch.Tensor | None = None
        self._splitk_seq_len: torch.Tensor | None = None
        self._splitk_sm_scale: float = 0.0
        self._use_splitk = False

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        *, skip_store: bool = False,
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)

        if not skip_store:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # Split-K attention for bs=1 decode (replaces FlashInfer)
        if self._use_splitk and q.shape[0] == 1:
            from minisgl.kernel.splitk_attention import splitk_attention_forward
            k_cache = self.kvcache.k_cache(layer_id).view(self.kvcache._storage_shape)
            v_cache = self.kvcache.v_cache(layer_id).view(self.kvcache._storage_shape)
            return splitk_attention_forward(
                q, k_cache, v_cache,
                self._splitk_page_indices,
                self._splitk_seq_len,
                self.qo_head_local, self.kv_head_local,
                self.config.head_dim, self._splitk_sm_scale,
                self._splitk_partial_out, self._splitk_partial_lse,
                self._splitk_out,
            )

        self._initialize_metadata_once(metadata)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def _init_decode_tensors(self, padded_size: int) -> None:
        """One-time init of pre-allocated pinned CPU tensors for decode path."""
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        self._decode_cu_seqlens_k_cpu = torch.zeros(padded_size + 1, **CPU_KWARGS)
        self._decode_seq_lens_cpu = torch.zeros(padded_size, **CPU_KWARGS)
        self._decode_cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        self._decode_cu_seqlens_q_gpu = self._decode_cu_seqlens_q_cpu.to(
            self.device, non_blocking=True
        )
        self._decode_last_page_cpu = torch.ones(padded_size, **CPU_KWARGS)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)

        # Fast path for decode: reuse pre-allocated tensors
        if batch.is_decode:
            if self._decode_cu_seqlens_k_cpu is None or len(self._decode_seq_lens_cpu) < padded_size:
                self._init_decode_tensors(padded_size)
            cu_seqlens_k = self._decode_cu_seqlens_k_cpu
            seq_lens = self._decode_seq_lens_cpu
            offset = 0
            for i, req in enumerate(reqs):
                sl = req.device_len
                seq_lens[i] = sl
                offset += sl
                cu_seqlens_k[i + 1] = offset
            page_table = get_global_ctx().page_table
            indices = torch.cat([page_table[req.table_idx, :req.device_len] for req in reqs])
            batch.attn_metadata = FIMetadata(
                cu_seqlens_q_cpu=self._decode_cu_seqlens_q_cpu[:padded_size + 1],
                cu_seqlens_k_cpu=cu_seqlens_k[:padded_size + 1],
                cu_seqlens_q_gpu=self._decode_cu_seqlens_q_gpu[:padded_size + 1],
                indices=indices,
                last_page_len_cpu=self._decode_last_page_cpu[:padded_size],
                num_qo_heads=self.qo_head_local,
                num_kv_heads=self.kv_head_local,
                head_dim=self.config.head_dim,
                page_size=1,
                pos_encoding_mode="NONE",
                seq_lens_cpu=seq_lens[:padded_size],
                dtype=self.kvcache.dtype,
                wrapper=self.decode_wrappers,
            )
            return

        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

        # Pre-allocate split-K attention buffers for bs=1 decode
        if 1 in bs_list:
            from minisgl.kernel.splitk_attention import _NUM_CHUNKS_PAD
            dev = self.kvcache.device
            hd = self.config.head_dim
            nqo = self.qo_head_local
            self._splitk_partial_out = torch.empty(nqo, _NUM_CHUNKS_PAD, hd, dtype=torch.float32, device=dev)
            self._splitk_partial_lse = torch.empty(nqo, _NUM_CHUNKS_PAD, dtype=torch.float32, device=dev)
            self._splitk_out = torch.empty(1, nqo, hd, dtype=self.kvcache.dtype, device=dev)
            self._splitk_seq_len = torch.zeros(1, dtype=torch.int32, device=dev)
            self._splitk_page_indices = capture.page_table  # pre-allocated indices buffer
            self._splitk_sm_scale = 1.0 / (hd ** 0.5)
            # Scratch buffer for normalized Q in fused qknorm+attn kernel
            self._splitk_q_scratch = torch.empty(nqo, hd, dtype=torch.float32, device=dev)
            # Atomic counter for last-block-reduces (persists across graph replays)
            self._splitk_head_counter = torch.zeros(nqo, dtype=torch.int32, device=dev)
            self._use_splitk = True

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture

        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)

        # Split-K path for bs=1: set up buffers, skip FlashInfer wrapper
        if self._use_splitk and bs == 1:
            seq_len = metadata.seq_lens_cpu[0].item()
            self._splitk_page_indices[:seq_len].copy_(metadata.indices[:seq_len])
            self._splitk_seq_len[0] = seq_len
            metadata.initialized = True
            self.graph_wrappers[bs] = None  # placeholder
            return

        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs

        # Split-K path: update page indices and seq_len directly, skip FlashInfer plan()
        if self._use_splitk and bs == 1:
            metadata.initialized = True
            # Copy fresh page indices into the pre-allocated buffer
            seq_len = metadata.seq_lens_cpu[0].item()
            self._splitk_page_indices[:seq_len].copy_(metadata.indices[:seq_len], non_blocking=True)
            self._splitk_seq_len[0] = seq_len
            return

        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_fast(metadata)

    def prepare_for_replay_multistep(
        self, batch: Batch, step_out_locs: torch.Tensor, num_steps: int
    ) -> None:
        """Set up split-K buffers for multi-step graph replay."""
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self._use_splitk and batch.padded_size == 1
        metadata.initialized = True

        seq_len = metadata.seq_lens_cpu[0].item()
        self._splitk_page_indices[:seq_len].copy_(
            metadata.indices[:seq_len], non_blocking=True
        )
        self._splitk_seq_len[0] = seq_len

        # Pre-write page indices for steps 1..num_steps-1
        # Step k's attention reads page_indices[0:seq_len+k], so page_indices[seq_len+k-1]
        # must hold the physical page for that position = step_out_locs[k]
        if num_steps > 1:
            self._splitk_page_indices[seq_len : seq_len + num_steps - 1].copy_(
                step_out_locs[1:num_steps]
            )

    def _initialize_metadata_fast(self, metadata: FIMetadata) -> None:
        """Like _initialize_metadata_once but skips the blocking synchronize."""
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # Skip self.last_event.synchronize() — safe for CUDA graph decode
        # because stream ordering ensures previous H2D copy completes before
        # the next plan's H2D copy starts on the same engine stream.
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

