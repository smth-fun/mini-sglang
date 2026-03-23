from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)

NUM_MULTI_STEPS = 4


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    next_tokens: torch.Tensor  # argmax result captured in graph
    # Multi-step buffers (bs=1 only)
    step_out_locs: torch.Tensor  # [NUM_MULTI_STEPS] pre-computed out_locs for each step
    step_tokens: torch.Tensor    # [NUM_MULTI_STEPS] argmax results for each step

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            next_tokens=torch.empty(bs, dtype=torch.int32, device=device),
            step_out_locs=torch.zeros(NUM_MULTI_STEPS, dtype=torch.int32, device=device),
            step_tokens=torch.empty(NUM_MULTI_STEPS, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.multistep_graph: torch.cuda.CUDAGraph | None = None
        self.has_multistep = False
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                self.buffer.next_tokens[:bs] = torch.argmax(self.buffer.logits[:bs], dim=-1).to(torch.int32)
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                    self.buffer.next_tokens[:bs] = torch.argmax(self.buffer.logits[:bs], dim=-1).to(torch.int32)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        # Capture multi-step graph for bs=1 greedy decode
        if 1 in self.graph_bs_list and self.attn_backend._use_splitk:
            self._capture_multistep(model, pool)

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_multistep(self, model: BaseLLMModel, pool) -> None:
        """Capture a CUDA graph with NUM_MULTI_STEPS sequential forward passes for bs=1."""
        N = NUM_MULTI_STEPS
        batch = Batch(reqs=[self.dummy_req], phase="decode")
        batch.padded_reqs = batch.reqs
        self.buffer.set_batch(batch)

        # Save and reset split-K seq_len for warmup
        saved_seq_len = self.attn_backend._splitk_seq_len[0].clone()

        with get_global_ctx().forward_batch(batch):
            # Warmup run (triggers Triton JIT if needed)
            for step in range(N):
                if step > 0:
                    self.buffer.input_ids[0] = self.buffer.step_tokens[step - 1]
                    self.buffer.positions[0] += 1
                    self.buffer.out_loc[0] = self.buffer.step_out_locs[step]
                    self.attn_backend._splitk_seq_len[0] += 1
                self.buffer.logits[:1] = model.forward()
                self.buffer.step_tokens[step] = torch.argmax(
                    self.buffer.logits[:1], dim=-1
                ).to(torch.int32)

        # Reset state for capture
        self.attn_backend._splitk_seq_len.copy_(saved_seq_len)
        self.buffer.set_batch(batch)

        graph = torch.cuda.CUDAGraph()
        with get_global_ctx().forward_batch(batch):
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                for step in range(N):
                    if step > 0:
                        self.buffer.input_ids[0] = self.buffer.step_tokens[step - 1]
                        self.buffer.positions[0] += 1
                        self.buffer.out_loc[0] = self.buffer.step_out_locs[step]
                        self.attn_backend._splitk_seq_len[0] += 1
                    self.buffer.logits[:1] = model.forward()
                    self.buffer.step_tokens[step] = torch.argmax(
                        self.buffer.logits[:1], dim=-1
                    ).to(torch.int32)

        self.multistep_graph = graph
        self.has_multistep = True
        logger.info_rank0(f"Captured multi-step CUDA graph with {N} steps")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> tuple:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size], self.buffer.next_tokens[: batch.size]

    def replay_multistep(self, batch: Batch, step_out_locs: torch.Tensor) -> tuple:
        """Replay the multi-step CUDA graph (4 forward passes in one replay)."""
        assert self.has_multistep and batch.size == 1
        self.buffer.copy_from(batch)
        self.buffer.step_out_locs[:NUM_MULTI_STEPS].copy_(step_out_locs)
        self.attn_backend.prepare_for_replay_multistep(
            batch, step_out_locs, NUM_MULTI_STEPS
        )
        self.multistep_graph.replay()
        return self.buffer.logits[:1], self.buffer.step_tokens[:NUM_MULTI_STEPS]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
