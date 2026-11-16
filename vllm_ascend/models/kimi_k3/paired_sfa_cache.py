# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
DSpark shares full-history block IDs when its sharding matches the target,
with a separate region in the backing for every draft layer.
"""

from dataclasses import dataclass, replace

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
else:
    from vllm.v1.core.kv_cache_planning import may_override_num_blocks


@dataclass
class PairedSFAGroup(KVCacheGroup):
    # Draft pages keep private placement even when they share target block IDs.
    draft_layer_names: frozenset[str] = frozenset()


def _draft_layer_specs(groups):
    """Return local draft specs; empty PP-local groups allocate no tensors."""
    return {
        name: spec
        for group in groups
        if isinstance(group, PairedSFAGroup)
        for name, spec in group.kv_cache_specs.items()
        if name in group.draft_layer_names
    }


def _validate_paired_sfa_pages(specs) -> None:
    """Check the page geometry settled by NPUPlatform before spec creation."""
    main = [s for s in specs.values() if isinstance(s, AscendMLAAttentionSpec)]
    recurrent = [s for s in specs.values() if isinstance(s, MambaSpec)]
    if not main:
        raise ValueError("A paired SFA cache requires main MLA specs.")
    page = max(s.real_page_size_bytes for s in main)
    if any(s.real_page_size_bytes != page for s in main):
        raise ValueError("Paired SFA main pages must have identical geometry.")
    if any(replace(s, page_size_padded=None).page_size_bytes > page for s in recurrent):
        raise ValueError("The MLA token block must cover one recurrent checkpoint.")
    if any(s.page_size_bytes != page for s in recurrent):
        raise ValueError("Paired SFA recurrent pages must match the main MLA page size.")


def paired_sfa_cache_groups(config, layer_configs: dict[str, KVCacheLayerConfig]) -> list[KVCacheGroup]:
    target = {name: layer for name, layer in layer_configs.items() if not layer.is_draft}
    draft = {name: layer for name, layer in layer_configs.items() if layer.is_draft}
    groups = _paired_sfa_target_cache_groups(config, target)
    if not draft:
        return groups
    if any(
        isinstance(layer.spec, AscendMLAAttentionSpec) and layer.spec.uses_packed_sfa_main_cache
        for layer in draft.values()
    ):
        raise ValueError("Kimi DSpark draft groups require ordinary attention cache specs.")
    full = groups[0]
    ordered = dict(sorted(draft.items(), key=lambda item: extract_layer_index(item[0])))
    separate = {name: layer for name, layer in ordered.items() if layer.spec.dcp_sharded != full.dcp_sharded}
    shared_draft = {name: layer for name, layer in ordered.items() if name not in separate}
    shared = {**{name: target[name] for name in full.layer_names}, **shared_draft}
    if any(layer.spec.tokens_per_state != 1 for layer in shared.values()):
        raise ValueError("Shared DSpark ownership requires uncompressed full-history token slots.")
    groups[0] = PairedSFAGroup.from_layers(
        shared,
        draft_layer_names=frozenset(shared_draft),
        is_eagle_group=full.is_eagle_group
        or any(getattr(layer.spec, "non_causal_multi_token_decode", False) for layer in shared_draft.values()),
    )
    if separate:
        groups.append(
            PairedSFAGroup.from_layers(
                separate,
                draft_layer_names=frozenset(separate),
                is_eagle_group=any(
                    getattr(layer.spec, "non_causal_multi_token_decode", False) for layer in separate.values()
                ),
            )
        )
    return groups


def _paired_sfa_target_cache_groups(config, layer_configs):
    specs = {name: layer.spec for name, layer in layer_configs.items()}
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Paired SFA cache requires the hybrid KV cache manager.")
    _validate_paired_sfa_pages(specs)
    main = sorted((n for n, s in specs.items() if isinstance(s, AscendMLAAttentionSpec)), key=extract_layer_index)
    indexer = sorted((n for n, s in specs.items() if isinstance(s, AscendSFAIndexerCacheSpec)), key=extract_layer_index)
    if not main or [extract_layer_index(n) for n in main] != [extract_layer_index(n) for n in indexer]:
        raise ValueError("Every sparse MLA layer must own one matching indexer cache.")
    paired = {n: specs[n] for pair in zip(main, indexer) for n in pair}
    if len({s.block_size for s in paired.values()}) != 1:
        raise ValueError("Paired MLA and indexer must use the same local token block size.")
    recurrent = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    recurrent = {n: recurrent[n] for n in sorted(recurrent, key=extract_layer_index)}
    # Each recurrent group overlays at most one main-page slot per target MLA
    # layer. Indexer and draft pages must not enlarge that physical region.
    groups = make_kimi_cache_groups(
        {name: layer_configs[name] for name in paired},
        {name: layer_configs[name] for name in recurrent},
        recurrent_group_size=len(main),
    )
    if len(paired) + len(recurrent) != len(specs):
        raise ValueError("Unexpected cache role in paired SFA layout.")
    return groups


def _layout(groups):
    full = next(g for g in groups if any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in g.distinct_specs))
    specs = full.kv_cache_specs
    # Draft layers share logical IDs but must not enlarge the target/KDA
    # overlay or inherit the target's packed page geometry. Retain global
    # specs here: a PP-local group can have no local target MLA layer.
    draft = _draft_layer_specs([full])
    main = [name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec) and name not in draft]
    indexer = [n for n in full.layer_names if isinstance(specs[n], AscendSFAIndexerCacheSpec)]
    recurrent = [g for g in groups if all(isinstance(spec, MambaSpec) for spec in g.distinct_specs)]
    main_page = next(s.page_size_bytes for s in full.distinct_specs if isinstance(s, AscendMLAAttentionSpec))
    indexer_page = next(s.page_size_bytes for s in full.distinct_specs if isinstance(s, AscendSFAIndexerCacheSpec))
    slots = max([len(main), *(len(g.layer_names) for g in recurrent)])
    return full, main, indexer, recurrent, main_page, indexer_page, slots


def paired_sfa_pool_bytes(groups):
    _, _, indexer, _, main_page, indexer_page, slots = _layout(groups)
    # Every pool ID still reserves independent target/indexer/draft storage.
    draft_bytes = sum(spec.page_size_bytes for spec in _draft_layer_specs(groups).values())
    return slots * main_page + len(indexer) * indexer_page + draft_bytes


def paired_sfa_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroup],
    available_memory: int,
    *,
    num_blocks: int | None = None,
) -> KVCacheConfig:
    _, main_mla_names, indexer_names, kda_groups, main_page_size, indexer_page_size, num_main_slots = _layout(
        kv_cache_groups
    )
    pool_bytes = paired_sfa_pool_bytes(kv_cache_groups)
    if num_blocks is None:
        num_blocks = may_override_num_blocks(vllm_config, available_memory // pool_bytes)
    backing_size = pool_bytes * num_blocks
    main_layer_size = main_page_size * num_blocks
    # Recurrent groups overlay the target MLA region. Their block IDs are
    # disjoint, while indexer and draft layers always own separate regions.
    tensors = [
        KVCacheTensor(
            size=backing_size,
            layers=main_mla_names,
            offset=0,
            layer_stride=main_layer_size,
            block_stride=main_page_size,
        )
    ]
    for group in kda_groups:
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=group.layer_names,
                offset=0,
                layer_stride=main_layer_size,
                block_stride=main_page_size,
            )
        )
    offset = num_main_slots * main_layer_size
    indexer_layer_size = indexer_page_size * num_blocks
    tensors.append(
        KVCacheTensor(
            size=backing_size,
            layers=indexer_names,
            offset=offset,
            layer_stride=indexer_layer_size,
            block_stride=indexer_page_size,
        )
    )
    offset += len(indexer_names) * indexer_layer_size
    for name, spec in _draft_layer_specs(kv_cache_groups).items():
        draft_layer_size = spec.page_size_bytes * num_blocks
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[name],
                offset=offset,
                layer_stride=draft_layer_size,
                block_stride=spec.page_size_bytes,
            )
        )
        offset += draft_layer_size
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def paired_sfa_max_memory(config, groups):
    blocks = sum(g.max_allocated_blocks_per_request(config) for g in groups)
    return blocks * paired_sfa_pool_bytes(groups)
