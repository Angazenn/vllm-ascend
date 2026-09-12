# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata and cache ownership for generalized nano Q1/MTP offload.

Registered host KV supplies sparse misses and bounded circular HBM tails.
Colocated execution may additionally retain a full device cache for prefill.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

TOPK = 2048
COPY_MISS_CAPACITY = 32768
LIM_MISS_CAPACITY = COPY_MISS_CAPACITY
INVALID_SLOT = -(1 << 31)
REQUEST_STATE_NON_OFFLOAD = -3
REQUEST_STATE_FIRST_OFFLOAD = -2
REQUEST_STATE_STEADY = -1


def prepare_copy_sfa_queries(query, query_rope):
    """Prepare contiguous queries for native 8/16/32/64/128-head tiles.

    MLA shares KV across query heads, and softmax is independent per head.
    Zero-filled extra query heads cannot affect the original heads; callers
    must return only the original head range from the kernel output.
    """
    heads = query.shape[1]
    if 1 <= heads < 8:
        padded_query = query.new_zeros((query.shape[0], 8, query.shape[2]))
        padded_rope = query_rope.new_zeros((query_rope.shape[0], 8, query_rope.shape[2]))
        padded_query[:, :heads].copy_(query)
        padded_rope[:, :heads].copy_(query_rope)
        return padded_query, padded_rope
    if heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            f"Generalized copy-SFA serving requires 1–8, 16, 32, 64 or 128 query heads per rank, got {heads}"
        )
    return query.contiguous(), query_rope.contiguous()


@dataclass
class MtpBatch:
    query_ends: torch.Tensor
    seq_lens: torch.Tensor
    offload_lens: torch.Tensor
    cache_tokens: torch.Tensor
    pool_entries: torch.Tensor
    hbm_block_table: torch.Tensor
    source_block_table: torch.Tensor
    tail_sources: torch.Tensor
    tail_destinations: torch.Tensor
    pool_rows: list[int]
    prefix_lengths: list[int]
    cache_sizes: list[int]
    num_tokens: int
    query_ends_cpu: tuple[int, ...]
    seq_lens_cpu: tuple[int, ...]
    graph_buffers: object | None = None


def _host_snapshot(values, tensor, count):
    # Standalone operator callers may only supply device metadata. Serving
    # passes exact CPU mirrors and never takes this compatibility readback.
    if values is None:
        values = tensor[:count].cpu()
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError("MTP host metadata must be on CPU")
        values = values[:count].tolist()
    snapshot = tuple(int(value) for value in values[:count])
    if len(snapshot) != count:
        raise ValueError("MTP host metadata is shorter than the request batch")
    return snapshot


def make_mtp_batch(
    metadata,
    manager,
    *,
    query_ends_cpu: Sequence[int] | torch.Tensor | None = None,
    seq_lens_cpu: Sequence[int] | torch.Tensor | None = None,
    pool_rows_cpu: Sequence[int] | torch.Tensor | None = None,
) -> MtpBatch | None:
    """Snapshot scheduled lengths, including rejection-adjusted drafts."""
    if metadata.num_prefills or not metadata.num_decodes:
        return None
    count = metadata.num_decodes
    # Builders reuse these buffers for target and draft steps. Keep an owned
    # snapshot: .to().contiguous() can retain a view whose values later change
    # while this batch's Python lengths and queued custom calls stay fixed.
    query_ends = (
        metadata.cum_query_lens[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    ends = _host_snapshot(query_ends_cpu, query_ends, count)
    widths = [end - start for start, end in zip((0,) + ends[:-1], ends)]
    seq_lens = (
        metadata.seq_lens[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    lengths = _host_snapshot(seq_lens_cpu, seq_lens, count)
    block_size = manager.block_size
    prefixes = [(length - width) // block_size * block_size for length, width in zip(lengths, widths)]
    if any(width < 1 or width > 7 for width in widths):
        return None
    if any(prefix < TOPK for prefix in prefixes):
        return None
    caches = [min(prefix, manager.topk_buffer_size) for prefix in prefixes]
    for width, prefix, cache in zip(widths, prefixes, caches):
        if (prefix <= width * TOPK and cache != prefix) or (
            prefix > width * TOPK and not width * TOPK <= cache <= 16256
        ):
            raise ValueError(f"Invalid generalized LIM budget: Q={width}, L={prefix}, C={cache}")
    pools = (
        metadata.req_topk_buffer_slots[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    pool_rows = list(_host_snapshot(pool_rows_cpu, pools, count))
    if len(set(pool_rows)) != count or any(pool < 0 or pool >= manager.max_num_reqs for pool in pool_rows):
        raise ValueError("Generalized MTP requests must occupy distinct pool rows")
    device = seq_lens.device
    source_block_table = metadata.block_table[:count].clone(memory_format=torch.contiguous_format)
    stride_blocks = manager.topk_buffer_size // block_size + 2
    table = torch.zeros((count, stride_blocks), dtype=torch.int32)
    source_rows, source_positions, destinations = [], [], []
    for row, (pool, prefix, length, cache) in enumerate(zip(pool_rows, prefixes, lengths, caches)):
        base = pool * stride_blocks
        cache_blocks = cache // block_size
        table[row, :cache_blocks] = torch.arange(cache_blocks, dtype=torch.int32) + base
        tail_start = base + stride_blocks - 2
        for tail_block in range(2):
            table[row, cache_blocks + tail_block] = tail_start + (prefix // block_size + tail_block) % 2
        for position in range(prefix, length):
            source_rows.append(row)
            source_positions.append(position)
            destinations.append(
                pool * stride_blocks * block_size + manager.topk_buffer_size + position % (2 * block_size)
            )
    positions = torch.tensor(source_positions, dtype=torch.int64, device=device)
    rows = torch.tensor(source_rows, dtype=torch.int64, device=device)
    source_blocks = source_block_table[rows, positions // block_size].to(torch.int64)
    return MtpBatch(
        query_ends,
        seq_lens,
        torch.tensor(prefixes, dtype=torch.int32, device=device),
        torch.tensor(caches, dtype=torch.int32, device=device),
        pools,
        table.to(device),
        source_block_table,
        source_blocks * block_size + positions % block_size,
        torch.tensor(destinations, dtype=torch.int64, device=device),
        pool_rows,
        prefixes,
        caches,
        ends[-1],
        ends,
        lengths,
    )


class EagerMtpBuffers:
    """Stream-owned LIM scratch and pinned uploads for sequential eager calls.

    All LIM and shared copy-SFA consumers must be enqueued before the next
    prepare on this stream, matching GeneralizedMtpRuntime's existing output
    lifetime. Stream order protects device scratch; upload events separately
    protect pinned memory from being overwritten by the CPU.
    """

    def __init__(self, requests, tokens, device):
        self.states = torch.empty(requests, dtype=torch.int32, device=device)
        self.outputs = tuple(
            torch.empty(shape, dtype=torch.int32, device=device)
            for shape in (
                (tokens, 1, TOPK),
                (tokens, 1, TOPK),
                (tokens,),
                (requests, LIM_MISS_CAPACITY),
                (requests, LIM_MISS_CAPACITY),
                (requests,),
            )
        )
        self.uploads = deque()

    def upload_states(self, states, stream):
        if self.uploads and self.uploads[0][2].query():
            staging, values, done = self.uploads.popleft()
        else:
            # Eager execution can enqueue several drafts while earlier work
            # is still running. Grow to the number of in-flight uploads rather
            # than synchronizing the host when a fixed ring would wrap.
            staging = torch.empty(self.states.shape, dtype=torch.int32, device="cpu", pin_memory=True)
            values = staging.numpy()
            done = torch.npu.Event()
        values.fill(REQUEST_STATE_NON_OFFLOAD)
        values[: len(states)] = states
        self.states.copy_(staging, non_blocking=True)
        done.record(stream)
        self.uploads.append((staging, values, done))
        return self.states[: len(states)]

    def output_views(self, requests, tokens):
        return tuple(tensor[: tokens if index < 3 else requests] for index, tensor in enumerate(self.outputs))


class GeneralizedMtpRuntime:
    def __init__(self, manager):
        self.manager = manager
        self.maps = {}
        self.residents = {}
        self.outputs = None
        self.output_batch = None
        self.eager_buffers = {}

    def invalidate(self):
        # A prefill or mixed step may rewrite cached tokens or recycle rows.
        self.residents.clear()
        self.output_batch = None

    def prepare_lim(self, layer_name, batch, source_capacity, device):
        if batch.graph_buffers is not None:
            return batch.graph_buffers.lim_inputs(layer_name)
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Captured LIM calls require prepared graph buffers")
        manager = self.manager
        mapping = self.maps.get(layer_name)
        if mapping is None:
            mapping = torch.full(
                (getattr(manager, "max_num_topk_rows", manager.max_num_reqs), source_capacity),
                INVALID_SLOT,
                dtype=torch.int32,
                device=device,
            )
            self.maps[layer_name] = mapping
        if mapping.shape[1] != source_capacity:
            raise ValueError("Generalized LIM source capacity changed after cache allocation")
        owners = {slot: req for req, slot in manager.topk_buffer_slot_manager.req2slot.items()}
        resident = self.residents.setdefault(layer_name, {})
        states = []
        for pool, prefix, cache in zip(batch.pool_rows, batch.prefix_lengths, batch.cache_sizes):
            owner = (owners[pool], manager.nano_mtp_slot_generations.get(pool, 0))
            previous = resident.get(pool)
            ready = previous is not None and previous[:2] == (owner, cache) and previous[2] <= prefix
            states.append(REQUEST_STATE_STEADY if ready else REQUEST_STATE_FIRST_OFFLOAD)
            resident[pool] = (owner, cache, prefix)
        count, tokens = len(batch.pool_rows), batch.num_tokens
        stream = torch.npu.current_stream(device)
        key = (stream.device, stream.npu_stream)
        buffers = self.eager_buffers.get(key)
        if buffers is None:
            buffers = EagerMtpBuffers(
                max(count, manager.max_num_reqs),
                max(tokens, manager.max_num_reqs * 7, getattr(manager, "max_num_topk_rows", 0)),
                device,
            )
            self.eager_buffers[key] = buffers
        if buffers.states.numel() < count or buffers.outputs[0].shape[0] < tokens:
            raise ValueError("Eager LIM batch exceeds its configured request/query capacity")
        state_tensor = buffers.upload_states(states, stream)
        self.outputs = buffers.output_views(count, tokens)
        self.output_batch = batch
        return mapping, state_tensor, self.outputs

    def require_outputs(self, batch):
        if batch.graph_buffers is not None:
            return batch.graph_buffers.outputs
        if self.output_batch is not batch or self.outputs is None:
            raise RuntimeError("Shared MTP attention ran before its indexer owner for this batch")
        return self.outputs

    def copy_metadata(self, batch):
        # Both kernels use contiguous int32[B, 32768]. Shared-indexer
        # consumers read the same valid miss-count prefixes without a bridge.
        return self.require_outputs(batch)
