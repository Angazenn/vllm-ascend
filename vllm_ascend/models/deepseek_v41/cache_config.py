# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache placements within a shared allocation."""

from collections.abc import Iterable
from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec, AscendMLAAttentionSpec
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)

STATE_RING_ROWS = 32

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_layout(*, block_size, head_size, dtype, cache_dtype):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * get_dtype_size(torch.bfloat16)
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
    )


def is_v41_swa_layout(spec):
    # V4.1 uses the plain MLA layout for SWA and Ascend MLA layouts for
    # compressed and indexer bytes.
    return type(spec) is MLAAttentionSpec and spec.model_version == "deepseek_v41"


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * get_dtype_size(torch.bfloat16)
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
    )


def is_deepseek_v41_cache(specs: Iterable[KVCacheSpec]) -> bool:
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted(
        (name for name, spec in specs.items() if isinstance(spec, AscendCircularBufferSpec)), key=_layer_number
    )
    swa = {name for name, spec in specs.items() if is_v41_swa_layout(spec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    # A reduced backbone may contain only the initial SWA layers. Without
    # full-context source slots, each SWA layer needs its own physical slot.
    if not full and len(swa) == len(specs):
        names = target_swa + draft_swa
        return [specs[name].unpadded_page_size_bytes for name in names], [(name,) for name in names]

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            [kv_bytes + index_bytes + folded_bytes, *(specs[name].unpadded_page_size_bytes for name in aliases)]
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def _group_layer_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if specs and all(is_v41_swa_layout(spec) for spec in specs.values()):
        return [{name: spec} for name, spec in specs.items()]
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        kv_name, index_name, *rest = layer_tuple
        folded_name = index_name + "_folded"
        has_folded = folded_name in rest
        aliases = rest[1:] if has_folded else rest
        kv_bytes = specs[kv_name].unpadded_page_size_bytes
        index_bytes = specs[index_name].unpadded_page_size_bytes
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        padded[index_name] = replace(
            specs[index_name], page_size_padded=index_bytes if has_folded else page_size - kv_bytes
        )
        if has_folded:
            padded[folded_name] = replace(specs[folded_name], page_size_padded=page_size - kv_bytes - index_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, AscendCircularBufferSpec)]
    groups = [
        {name: padded[name] for name in mla_names},
        {name: padded[name] for name in state_names},
    ]

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    swa_columns = [
        [name for name in layer_tuple if is_v41_swa_layout(padded[name]) and ".mtp." not in f".{name}"]
        for layer_tuple in layer_tuples
    ]
    for names in zip(*swa_columns):
        groups.append({name: padded[name] for name in names})

    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if is_v41_swa_layout(padded[name]) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append({name: padded[name] for name in draft_names})
    return groups


def group_layer_configs(layer_configs: dict[str, KVCacheLayerConfig]) -> list[KVCacheGroup]:
    specs = {name: config.spec for name, config in layer_configs.items()}
    return [
        KVCacheGroup.from_layers({name: replace(layer_configs[name], spec=spec) for name, spec in group.items()})
        for group in _group_layer_specs(specs)
        if group
    ]


def place_deepseek_v41_cache(vllm_config: VllmConfig, groups: list[KVCacheGroup], num_blocks: int) -> KVCacheConfig:
    """Place resource pages in layer slots within one shared backing pool."""
    specs = {name: spec for group in groups for name, spec in group.kv_cache_specs.items()}
    page_sizes, layer_tuples = get_layer_tuples(specs)
    backing_size = num_blocks * sum(page_sizes)
    tensors: list[KVCacheTensor] = []
    slot_offset = 0
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        # Long KV, split indexer and optional folded indexer are consecutive
        # within a page. State/SWA groups reuse that page from its beginning.
        component_offset = 0
        for name in layer_names:
            spec = specs[name]
            is_mla = isinstance(spec, AscendMLAAttentionSpec)
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_offset + (component_offset if is_mla else 0),
                    layer_stride=num_blocks * page_size,
                    block_stride=page_size,
                )
            )
            if is_mla:
                component_offset += spec.unpadded_page_size_bytes
        slot_offset += num_blocks * page_size
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
