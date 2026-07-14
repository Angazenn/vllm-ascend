# SPDX-License-Identifier: Apache-2.0
"""Physical cache planning and tuple slots for SFA decode offload."""

from dataclasses import dataclass
import re
from typing import Any

from vllm.utils.torch_utils import get_dtype_size

from vllm_ascend.core.kv_cache_interface import (
    AscendSFAOffloadIndexerCacheSpec,
    OffloadMLAAttentionSpec,
)

# BF16 direct decode offload seven-tuple:
#   [0] shared main K scratch
#   [1] shared main V scratch
#   [2] real owner indexer alias
#   [3] per-layer resident K
#   [4] per-layer resident V
#   [5] per-layer/request two-block tail K
#   [6] per-layer/request two-block tail V
#
# The existing sparse-C8 path remains a six-tuple with indexer scale at [5].
OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4
OFFLOAD_TAIL_K = 5
OFFLOAD_TAIL_V = 6
OFFLOAD_INDEXER_S = 5

OFFLOAD_TUPLE_LEN = 7
OFFLOAD_C8_TUPLE_LEN = 6
OFFLOAD_LEGACY_TUPLE_LEN = 5
TAIL_WINDOW_BLOCKS = 2

_SFA_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_ALIGNMENT_BYTES = 2 * 1024 * 1024


def is_offload_c8_kv_cache(kv_cache: tuple) -> bool:
    return len(kv_cache) == OFFLOAD_C8_TUPLE_LEN


def is_direct_sfa_kv_offload(vllm_config: Any) -> bool:
    transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return (
        transfer_config is not None
        and getattr(transfer_config, "kv_connector", None)
        == "SFAKVOffloadConnector"
    )


def extract_sfa_layer_id(layer_name: str) -> int:
    match = _SFA_LAYER_RE.search(layer_name)
    if match is None:
        raise ValueError(
            f"Cannot extract an SFA layer id from {layer_name!r}."
        )
    return int(match.group(1))


@dataclass(frozen=True)
class SFAOffloadSharedCachePool:
    """One real indexer owner and its time-multiplexed main KV layers."""

    indexer_layer_name: str
    main_layer_names: tuple[str, ...]

    @property
    def shared_by(self) -> tuple[str, ...]:
        # Keep a main owner first so the model runner allocates the raw K/V
        # layout before binding the indexer alias.
        return (*self.main_layer_names, self.indexer_layer_name)


@dataclass(frozen=True)
class SFAOffloadSharedCachePlan:
    """Exact direct-offload layout shared by scheduler and model runner."""

    indexer_group_id: int
    main_group_id: int
    pools: tuple[SFAOffloadSharedCachePool, ...]
    main_page_bytes: int
    fixed_hbm_bytes: int
    alignment_reserve_bytes: int

    @property
    def num_physical_pools(self) -> int:
        return len(self.pools)

    @property
    def main_layer_names(self) -> tuple[str, ...]:
        return tuple(
            layer_name
            for pool in self.pools
            for layer_name in pool.main_layer_names
        )

    @property
    def indexer_layer_names(self) -> tuple[str, ...]:
        return tuple(pool.indexer_layer_name for pool in self.pools)

    @property
    def total_fixed_hbm_bytes(self) -> int:
        return self.fixed_hbm_bytes + self.alignment_reserve_bytes


def _resolve_direct_offload_groups(
    kv_cache_groups: list[Any],
) -> tuple[int, Any, int, Any] | None:
    indexer_groups = [
        (group_id, group)
        for group_id, group in enumerate(kv_cache_groups)
        if isinstance(
            group.kv_cache_spec,
            AscendSFAOffloadIndexerCacheSpec,
        )
    ]
    main_groups = [
        (group_id, group)
        for group_id, group in enumerate(kv_cache_groups)
        if isinstance(group.kv_cache_spec, OffloadMLAAttentionSpec)
    ]
    # Legacy direct/C8 layouts also use OffloadMLAAttentionSpec but do not
    # expose the owner-indexer marker. Leave those layouts untouched.
    if not indexer_groups:
        return None
    if len(indexer_groups) != 1 or len(main_groups) != 1:
        raise ValueError(
            "Direct SFA decode offload requires exactly one real-indexer "
            "group and one main-KV group."
        )
    if len(kv_cache_groups) != 2:
        raise ValueError(
            "Direct SFA decode offload does not support additional KV cache "
            f"groups, got {len(kv_cache_groups)}."
        )
    indexer_group_id, indexer_group = indexer_groups[0]
    main_group_id, main_group = main_groups[0]
    return indexer_group_id, indexer_group, main_group_id, main_group


def build_sfa_offload_shared_cache_plan(
    vllm_config: Any,
    kv_cache_groups: list[Any],
    *,
    resident_capacity: int | None = None,
    alignment_bytes: int = _ALIGNMENT_BYTES,
) -> SFAOffloadSharedCachePlan | None:
    """Build owner-anchored BF16 HBM pools for direct decode offload."""

    if not is_direct_sfa_kv_offload(vllm_config):
        return None
    resolved = _resolve_direct_offload_groups(kv_cache_groups)
    if resolved is None:
        return None
    indexer_group_id, indexer_group, main_group_id, main_group = resolved

    main_by_id = {
        extract_sfa_layer_id(name): name
        for name in main_group.layer_names
    }
    indexer_by_id = {
        extract_sfa_layer_id(name): name
        for name in indexer_group.layer_names
    }
    if len(main_by_id) != len(main_group.layer_names):
        raise ValueError("Direct SFA offload has duplicate main layer ids.")
    if len(indexer_by_id) != len(indexer_group.layer_names):
        raise ValueError("Direct SFA offload has duplicate indexer layer ids.")
    if not main_by_id or not indexer_by_id:
        raise ValueError(
            "Direct SFA offload requires both main and real indexer layers."
        )
    if not set(indexer_by_id).issubset(main_by_id):
        raise ValueError(
            "Every direct SFA indexer owner must have a matching main layer; "
            f"extra owners={sorted(set(indexer_by_id) - set(main_by_id))}."
        )

    owner_ids = sorted(indexer_by_id)
    first_main_id = min(main_by_id)
    if owner_ids[0] > first_main_id:
        raise ValueError(
            "Direct SFA offload cannot assign main layers before the first "
            f"real indexer owner: first_main={first_main_id}, "
            f"first_owner={owner_ids[0]}."
        )

    mains_by_owner: dict[int, list[str]] = {
        owner_id: [] for owner_id in owner_ids
    }
    owner_index = 0
    for main_id in sorted(main_by_id):
        while (
            owner_index + 1 < len(owner_ids)
            and owner_ids[owner_index + 1] <= main_id
        ):
            owner_index += 1
        owner_id = owner_ids[owner_index]
        mains_by_owner[owner_id].append(main_by_id[main_id])

    pools = tuple(
        SFAOffloadSharedCachePool(
            indexer_layer_name=indexer_by_id[owner_id],
            main_layer_names=tuple(mains_by_owner[owner_id]),
        )
        for owner_id in owner_ids
    )
    empty_owners = [
        pool.indexer_layer_name for pool in pools if not pool.main_layer_names
    ]
    if empty_owners:
        raise ValueError(
            "Every direct SFA indexer owner must anchor at least one main "
            f"layer, empty owners={empty_owners}."
        )

    num_hidden_layers = getattr(
        vllm_config.model_config.hf_text_config,
        "num_hidden_layers",
        None,
    )
    if num_hidden_layers is not None:
        mtp_main_ids = {
            layer_id for layer_id in main_by_id if layer_id >= num_hidden_layers
        }
        missing_mtp_owners = mtp_main_ids - set(indexer_by_id)
        if missing_mtp_owners:
            raise ValueError(
                "Direct SFA offload requires each MTP main layer to own a real "
                f"indexer, missing layer ids={sorted(missing_mtp_owners)}."
            )

    main_spec = main_group.kv_cache_spec
    indexer_spec = indexer_group.kv_cache_spec
    if main_spec.page_size_bytes != indexer_spec.page_size_bytes:
        raise ValueError(
            "Direct SFA main and indexer manager pages must have equal byte "
            f"sizes, got main={main_spec.page_size_bytes}, "
            f"indexer={indexer_spec.page_size_bytes}."
        )

    scheduler_config = vllm_config.scheduler_config
    max_num_reqs = scheduler_config.max_num_seqs
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens
    max_num_topk_rows = min(
        scheduler_config.max_num_batched_tokens,
        max_num_reqs * decode_width,
    )
    if resident_capacity is None:
        # Capacity checks also run in EngineCore before the process-local
        # AscendConfig singleton is initialized. Read the serialized source
        # config so scheduler planning and worker allocation use the same
        # resident workspace size in every process.
        from vllm_ascend.ascend_config import LRUResidentCacheConfig

        additional_config = getattr(vllm_config, "additional_config", None)
        additional_config = additional_config or {}
        lru_config = LRUResidentCacheConfig(
            additional_config.get("lru_resident_cache_config", {})
        )
        resident_capacity = (
            lru_config.buffer_size if lru_config.enabled else 2048
        )

    dtype_bytes = get_dtype_size(main_spec.dtype)
    per_token_bytes = main_spec.head_size * dtype_bytes
    num_main_layers = len(main_by_id)
    resident_bytes = (
        num_main_layers
        * max_num_topk_rows
        * resident_capacity
        * per_token_bytes
    )
    tail_bytes = (
        num_main_layers
        * max_num_reqs
        * TAIL_WINDOW_BLOCKS
        * main_spec.page_size_bytes
    )
    alignment_reserve_bytes = len(pools) * 2 * alignment_bytes

    return SFAOffloadSharedCachePlan(
        indexer_group_id=indexer_group_id,
        main_group_id=main_group_id,
        pools=pools,
        main_page_bytes=main_spec.page_size_bytes,
        fixed_hbm_bytes=resident_bytes + tail_bytes,
        alignment_reserve_bytes=alignment_reserve_bytes,
    )
