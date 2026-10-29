# SPDX-License-Identifier: Apache-2.0
"""[Role] Plan and materialize Qwen3.5 MRV2 cache storage.

[Invariants] Preserve the upstream descriptor placements and one hybrid pool.
[Timeouts & Locks] No waits or locks; used only during worker initialization.
[Side-Effect Policy] Planning is pure; materialization allocates through its caller.

Planning lives here until Ascend can use upstream KVCacheConfigBuilder. The
plan adds allocation identity without changing upstream KVCacheTensor fields.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.attention.kv_cache_views import CacheViewPlacement

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class CacheAllocation:
    size_bytes: int


@dataclass(frozen=True)
class LayerCachePlan:
    allocation_index: int
    placement: CacheViewPlacement
    spec: "KVCacheSpec"


@dataclass(frozen=True)
class AscendCachePlan:
    allocations: tuple[CacheAllocation, ...]
    layers: Mapping[str, LayerCachePlan]
    shared_layers: Mapping[str, str]


def build_qwen3_5_cache_plan(
    config: "KVCacheConfig",
    layer_specs: Mapping[str, "KVCacheSpec"],
    shared_layers: Mapping[str, str],
) -> AscendCachePlan:
    """Translate the Qwen hybrid pool contract into explicit layer placements.

    All descriptors in this hybrid path address one pool by upstream contract;
    equal sizes validate that contract, rather than establish allocation identity.
    """
    if not config.kv_cache_tensors:
        return AscendCachePlan((), {}, dict(shared_layers))
    pool_size = config.kv_cache_tensors[0].size
    if any(tensor.size != pool_size or tensor.host_resident for tensor in config.kv_cache_tensors):
        raise ValueError("Qwen3.5 hybrid cache descriptors must address one device pool.")
    layers = {}
    for descriptor in config.kv_cache_tensors:
        for layer_index, name in enumerate(descriptor.layers):
            if name in shared_layers:
                continue
            spec = layer_specs[name]
            num_blocks = config.num_blocks_of(descriptor)
            offset = descriptor.offset + layer_index * descriptor.layer_stride
            end = offset + (num_blocks - 1) * descriptor.block_stride + spec.page_size_bytes
            if offset < 0 or descriptor.block_stride < spec.page_size_bytes or end > pool_size:
                raise ValueError(f"Cache placement for {name} exceeds its pool or overlaps adjacent pages.")
            layers[name] = LayerCachePlan(
                allocation_index=0,
                placement=CacheViewPlacement(num_blocks, offset, descriptor.block_stride),
                spec=spec,
            )
    if layers.keys() != layer_specs.keys() - shared_layers.keys():
        raise ValueError("Qwen3.5 cache plan must place every non-shared layer.")
    return AscendCachePlan((CacheAllocation(pool_size),), layers, dict(shared_layers))


def allocate_cache_storage(
    plan: AscendCachePlan,
    allocate: Callable[[int], torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Materialize each planned allocation once, retaining caller alignment policy."""
    return tuple(allocate(allocation.size_bytes) for allocation in plan.allocations)


def create_backend_cache_views(
    plan: AscendCachePlan,
    allocations: Sequence[torch.Tensor],
    attn_groups: Sequence["AttentionGroup"],
    kernel_block_sizes: Sequence[int],
) -> dict[str, Any]:
    """Delegate component interpretation to backends, then apply semantic aliases."""
    groups = {name: group for group in attn_groups for name in group.layer_names}
    caches = {
        name: groups[name].backend.get_kernel_cache_views(
            allocations[layer.allocation_index],
            layer.spec,
            kernel_block_size=kernel_block_sizes[groups[name].kv_cache_group_id],
            placement=layer.placement,
        )
        for name, layer in plan.layers.items()
    }
    for name, target in plan.shared_layers.items():
        caches[name] = caches[target]
    return caches
