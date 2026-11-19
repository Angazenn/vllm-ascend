# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
DSpark shares full-history block IDs when its sharding matches the target,
with a separate region in the backing for every draft layer.
"""

from dataclasses import replace
from itertools import pairwise

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs, may_override_num_blocks
else:
    from vllm.v1.core.kv_cache_planning import create_kv_cache_group_specs, may_override_num_blocks


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


def paired_sfa_cache_groups(config, specs):
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Paired SFA cache requires the hybrid KV cache manager.")
    # Worker-local profiling can see only KDA layers on a short PP stage.
    if specs and all(isinstance(spec, MambaSpec) for spec in specs.values()):
        return create_kv_cache_group_specs(specs, [list(specs)])
    draft = {name: spec for name, spec in specs.items() if spec.is_draft}
    target = {name: spec for name, spec in specs.items() if name not in draft}
    groups = _paired_sfa_target_cache_groups(target)
    if not draft:
        return groups
    if any(isinstance(spec, AscendMLAAttentionSpec) and spec.uses_packed_sfa_main_cache for spec in draft.values()):
        raise ValueError("Kimi DSpark draft groups require ordinary attention cache specs.")
    ordered = dict(sorted(draft.items(), key=lambda item: extract_layer_index(item[0])))

    # Share logical ownership while retaining separate physical regions.
    # UniformType retains per-layer dtype/layout/non-causal flags; the strict
    # AscendMLAAttentionSpec.merge must not merge these different MLA specs.
    full = groups[0]
    assert isinstance(full.kv_cache_spec, UniformTypeKVCacheSpecs)
    # Replicated and DCP-sharded attention have different logical coverage.
    # Preserve their own block tables while retaining independent physical
    # regions in the paired backing allocation.
    separate = {name: spec for name, spec in ordered.items() if spec.dcp_sharded != full.kv_cache_spec.dcp_sharded}
    ordered = {name: spec for name, spec in ordered.items() if name not in separate}
    if separate:
        uniform_draft = UniformTypeKVCacheSpecs.from_specs(separate)
        if uniform_draft is None:
            raise ValueError("Draft layers must share full-attention semantics and logical block size.")
        groups.append(
            KVCacheGroupSpec(
                list(separate),
                uniform_draft,
                is_eagle_group=any(getattr(spec, "non_causal_multi_token_decode", False) for spec in separate.values()),
            )
        )
    shared = {**full.kv_cache_spec.kv_cache_specs, **ordered}
    if any(spec.tokens_per_state != 1 for spec in shared.values()):
        raise ValueError("Shared DSpark ownership requires uncompressed full-history token slots.")
    uniform = UniformTypeKVCacheSpecs.from_specs(shared)
    if uniform is None:
        raise ValueError("Target and DSpark must share full-attention semantics and logical block size.")
    max_len = config.model_config.max_model_len
    widths = {spec.max_num_blocks_per_req(config, max_len) for spec in shared.values()}
    if len(widths) != 1:
        raise ValueError("Target and DSpark must have identical block-table coverage, including DCP sharding.")
    # Keep target first for scheduler conversion and keep recurrent group IDs.
    # One request now owns one full-history block sequence instead of two.
    # Retain drafter prefix-tail protection after scheduler spec unwrapping;
    # this flag does not change the target's causal attention metadata.
    groups[0] = replace(
        full,
        layer_names=list(shared),
        kv_cache_spec=uniform,
        is_eagle_group=full.is_eagle_group
        or any(getattr(spec, "non_causal_multi_token_decode", False) for spec in ordered.values()),
    )
    return groups


def _paired_sfa_target_cache_groups(specs):
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
        paired,
        recurrent,
        recurrent_group_size=len(main),
        uniform_recurrent=False,
    )
    if groups is None:
        raise ValueError("Paired SFA components must share full-attention block semantics.")
    if len(paired) + len(recurrent) != len(specs):
        raise ValueError("Unexpected cache role in paired SFA layout.")
    return groups


def _paired_sfa_regions(groups):
    """Return local tensor regions and their total size in bytes per block."""
    # PP owns whole decoder layers, so each local target MLA is immediately
    # followed by its indexer in the paired group. Draft layers follow those pairs.
    full = groups[0]
    regions = []
    indexers = []
    if isinstance(full.kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = full.kv_cache_spec.kv_cache_specs
        main = []
        for name, following in pairwise(full.layer_names):
            if isinstance(specs[following], AscendSFAIndexerCacheSpec):
                main.append(name)
                indexers.append(following)
        if main:
            regions.append((main, 0, specs[main[0]].page_size_bytes))

    # KDA pages are padded to the target MLA stride at spec creation; all
    # local recurrent groups can therefore overlay the backing at offset zero.
    regions.extend(
        (group.layer_names, 0, group.kv_cache_spec.page_size_bytes)
        for group in groups
        if group.layer_names and isinstance(group.kv_cache_spec, MambaSpec)
    )
    offset = max((len(names) * page for names, _, page in regions), default=0)
    if indexers:
        page = specs[indexers[0]].page_size_bytes
        regions.append((indexers, offset, page))
        offset += len(indexers) * page

    # Every remaining local layer owns an independent region using its own page size.
    placed = {name for names, _, _ in regions for name in names}
    for group in groups:
        group_spec = group.kv_cache_spec
        for name in group.layer_names:
            if name not in placed:
                spec = (
                    group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
                )
                regions.append(([name], offset, spec.page_size_bytes))
                offset += spec.page_size_bytes
    return regions, offset


def paired_sfa_pool_bytes(groups):
    _, pool_bytes = _paired_sfa_regions(groups)
    return pool_bytes


def paired_sfa_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    regions, pool_bytes = _paired_sfa_regions(kv_cache_groups)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // pool_bytes)
    tensors = [
        KVCacheTensor(
            size=pool_bytes * num_blocks,
            layers=layers,
            offset=offset * num_blocks,
            layer_stride=page_size * num_blocks,
            block_stride=page_size,
        )
        for layers, offset, page_size in regions
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def paired_sfa_max_memory(config, groups):
    blocks = sum(
        (g.kv_cache_spec.max_memory_usage_bytes(config) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )
    return blocks * paired_sfa_pool_bytes(groups)
