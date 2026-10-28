"""Single-backing allocation using upstream cache descriptors and views.

[Role] Bridge Ascend's aligned allocation to upstream logical cache views.
[Invariants] All descriptors address one backing; layers unpack their own pages.
[Timeouts & Locks] Local allocation and view construction; no locks or waits.
[Side-Effect Policy] Allocates through the caller and returns zero-copy views.
"""

from collections.abc import Callable, Iterable

import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs, create_kv_cache_views
from vllm.v1.kv_cache_layout import KVCacheLayout


def uses_upstream_cache_views(layers: Iterable[AttentionLayerBase]) -> bool:
    """A backend opts its cache plan into upstream allocation and layer binding."""
    return any(getattr(layer.get_attn_backend(), "uses_upstream_cache_views", False) for layer in layers)


def allocate_shared_cache(
    config: KVCacheConfig,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, torch.Tensor]:
    sizes = {descriptor.size for descriptor in config.kv_cache_tensors}
    (size,) = sizes
    backing = allocate(size)
    return {name: backing for descriptor in config.kv_cache_tensors for name in descriptor.layers}


def create_shared_cache_views(
    config: KVCacheConfig,
    raw: dict[str, torch.Tensor],
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
) -> dict[str, torch.Tensor]:
    groups = {
        name: (index, group.kv_cache_spec)
        for index, group in enumerate(config.kv_cache_groups)
        for name in group.layer_names
    }
    caches = {}
    for descriptor in config.kv_cache_tensors:
        first = descriptor.layers[0]
        index, spec = groups[first]
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[first]
        views = create_kv_cache_views(
            raw[first], spec, config.num_blocks, layout, descriptor, kernel_block_sizes[index]
        )
        caches.update(zip(descriptor.layers, views, strict=True))
    return caches
