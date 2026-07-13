# SPDX-License-Identifier: Apache-2.0
"""Semantic cache-group discovery for split-cache SFA PD offload."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFALayerwiseIndexerCacheSpec,
    AscendSFAOffloadIndexerCacheSpec,
    OffloadMLAAttentionSpec,
)

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def get_transformer_layer_id(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    if match is None:
        raise ValueError(
            f"No transformer layer index in KV cache name {layer_name!r}"
        )
    return int(match.group(1))


@dataclass(frozen=True)
class SFASplitCacheLayout:
    main_group_id: int
    indexer_group_ids: tuple[int, ...]
    main_layer_names: tuple[str, ...]
    indexer_layer_names: tuple[str, ...]
    layer_to_group_id: dict[str, int]

    @property
    def indexer_owner_set(self) -> frozenset[str]:
        return frozenset(self.indexer_layer_names)

    @property
    def main_name_by_layer_id(self) -> dict[int, str]:
        return {
            get_transformer_layer_id(name): name
            for name in self.main_layer_names
        }

    @property
    def indexer_name_by_layer_id(self) -> dict[int, str]:
        return {
            get_transformer_layer_id(name): name
            for name in self.indexer_layer_names
        }


def _group_specs(group) -> list[object]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def resolve_sfa_split_cache_layout(
    kv_cache_config: KVCacheConfig,
    side: Literal["prefill", "decode"],
) -> SFASplitCacheLayout:
    """Resolve main/indexer groups by their cache specs, never by position."""
    if side == "prefill":
        main_type = AscendMLAAttentionSpec
        indexer_type = AscendSFALayerwiseIndexerCacheSpec
    elif side == "decode":
        main_type = OffloadMLAAttentionSpec
        indexer_type = AscendSFAOffloadIndexerCacheSpec
    else:
        raise ValueError(f"Unsupported SFA split-cache side: {side!r}")

    main_group_ids: list[int] = []
    indexer_group_ids: list[int] = []
    layer_to_group_id: dict[str, int] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        specs = _group_specs(group)
        if specs and all(isinstance(spec, main_type) for spec in specs):
            main_group_ids.append(group_id)
        elif specs and all(isinstance(spec, indexer_type) for spec in specs):
            indexer_group_ids.append(group_id)
        else:
            raise ValueError(
                "SFA split-cache PD found an unsupported or mixed cache group "
                f"at index {group_id}: {[type(spec).__name__ for spec in specs]}"
            )
        for layer_name in group.layer_names:
            if layer_name in layer_to_group_id:
                raise ValueError(
                    f"SFA cache owner {layer_name!r} appears in multiple groups"
                )
            layer_to_group_id[layer_name] = group_id

    if len(main_group_ids) != 1:
        raise ValueError(
            "SFA split-cache PD requires exactly one main-KV group; "
            f"found {main_group_ids}"
        )
    if not indexer_group_ids:
        raise ValueError("SFA split-cache PD requires at least one real-indexer group")

    main_group_id = main_group_ids[0]
    main_names = sorted(
        kv_cache_config.kv_cache_groups[main_group_id].layer_names,
        key=get_transformer_layer_id,
    )
    indexer_names = sorted(
        (
            name
            for group_id in indexer_group_ids
            for name in kv_cache_config.kv_cache_groups[group_id].layer_names
        ),
        key=get_transformer_layer_id,
    )
    main_ids = {get_transformer_layer_id(name) for name in main_names}
    indexer_ids = {get_transformer_layer_id(name) for name in indexer_names}
    if not indexer_ids.issubset(main_ids):
        raise ValueError(
            "SFA real-indexer owners must be a subset of main layers; "
            f"extra layer ids={sorted(indexer_ids - main_ids)}"
        )
    if len(indexer_ids) != len(indexer_names):
        raise ValueError(
            "SFA split-cache PD requires at most one real indexer per "
            "transformer layer"
        )

    return SFASplitCacheLayout(
        main_group_id=main_group_id,
        indexer_group_ids=tuple(indexer_group_ids),
        main_layer_names=tuple(main_names),
        indexer_layer_names=tuple(indexer_names),
        layer_to_group_id=layer_to_group_id,
    )
