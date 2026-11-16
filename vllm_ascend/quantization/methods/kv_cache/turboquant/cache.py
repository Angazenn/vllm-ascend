# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack DeepSeek C4 TQ pages using upstream byte-offset cache descriptors.

Each physical slot has the same page stride for every group that shares it.
Block IDs can therefore never alias a different block ID in another group.
"""

import math

from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size

# Both supported versions expose planning helpers in kv_cache_planning.
from vllm.v1.kv_cache_interface import (
    KVCacheGroup,
    KVCacheTensor,
)

from . import SLOT_BYTES, TURBOQUANT_CACHE_DTYPE

# PA attention's fast copy path requires aligned page starts. Preserve the
# integral compact-row stride used by the scatter update path as well.
PHYSICAL_STRIDE_ALIGNMENT = 256


def uses_turboquant_groups(groups):
    return any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in groups
        for spec in group.distinct_specs
    )


def group_layers(grouped_layers):
    largest_page = max(config.spec.page_size_bytes for group in grouped_layers for config in group.values())
    c4_group_bytes = [
        sum(config.spec.page_size_bytes for config in group.values())
        for group in grouped_layers
        if any(
            getattr(config.spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
            and config.spec.head_size == SLOT_BYTES
            for config in group.values()
        )
    ]
    budget = round_up(max(c4_group_bytes, default=largest_page), largest_page)
    groups = []
    for layers in grouped_layers:
        chunk = {}
        used = 0
        for name, config in layers.items():
            size = config.spec.page_size_bytes
            if chunk and used + size > budget:
                groups.append(KVCacheGroup.from_layers(chunk))
                chunk, used = {}, 0
            chunk[name] = config
            used += size
        if chunk:
            groups.append(KVCacheGroup.from_layers(chunk))
    return groups


def _alignment(spec):
    alignment = get_dtype_size(spec.dtype)
    if getattr(spec, "scale_dim", 0):
        alignment = math.lcm(alignment, get_dtype_size(spec.scale_dtype))
    if getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE and spec.head_size == SLOT_BYTES:
        # The fused kernel computes the physical row stride from the page
        # stride. Preserve an integral number of compact 258-byte rows.
        alignment = math.lcm(alignment, SLOT_BYTES)
    return alignment


def packed_layout(groups):
    specs = {name: spec for group in groups for name, spec in group.kv_cache_specs.items()}
    page_alignment = math.lcm(PHYSICAL_STRIDE_ALIGNMENT, *(_alignment(spec) for spec in specs.values()))
    # A block ID belongs to one group at a time. Overlay complete groups
    # with a common stride to avoid fragmentation between small-page bins.
    offsets = {}
    page_size = 0
    for group in groups:
        used = 0
        for name in sorted(group.layer_names, key=lambda name: -specs[name].page_size_bytes):
            spec = specs[name]
            offset = round_up(used, _alignment(spec))
            offsets[name] = offset
            used = offset + spec.page_size_bytes
        page_size = max(page_size, used)
    return [(round_up(page_size, page_alignment), offsets)]


def pool_bytes_per_block(groups):
    return sum(size for size, _ in packed_layout(groups))


def cache_tensors(groups, num_blocks):
    layout = packed_layout(groups)
    backing_size = num_blocks * sum(size for size, _ in layout)
    descriptors = []
    slot_base = 0
    for stride, offsets in layout:
        for name, offset in offsets.items():
            descriptors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_base + offset,
                    layer_stride=0,
                    block_stride=stride,
                )
            )
        slot_base += stride * num_blocks
    return descriptors
