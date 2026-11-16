# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, KVCacheLayerConfig, MambaSpec, MLAAttentionSpec

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


def make_kimi_cache_groups(
    full_history: dict[str, KVCacheLayerConfig],
    recurrent: dict[str, KVCacheLayerConfig],
    *,
    recurrent_group_size: int,
) -> list[KVCacheGroup]:
    """Share full-history IDs and balance recurrent groups by slot budget."""
    names = list(recurrent)
    count = cdiv(len(names), recurrent_group_size)
    return [
        KVCacheGroup.from_layers(full_history),
        *(KVCacheGroup.from_layers({name: recurrent[name] for name in names[i::count]}) for i in range(count)),
    ]


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroup] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    target_attention_specs = {
        name: config
        for name, config in layer_configs.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX)
        and config.manager_cls is FullAttentionManager
        and isinstance(config.spec, AttentionSpec)
        and not isinstance(config.spec, MLAAttentionSpec)
    }
    draft_attention_specs = {
        name: config
        for name, config in layer_configs.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX)
        and config.manager_cls is FullAttentionManager
        and isinstance(config.spec, AttentionSpec)
        and not isinstance(config.spec, MLAAttentionSpec)
    }
    mamba_specs = {
        name: config
        for name, config in layer_configs.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(config.spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(layer_configs)
    ):
        return None

    attention_configs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    attention_specs = [config.spec for config in attention_configs]
    all_specs = [*attention_specs, *(config.spec for config in mamba_specs.values())]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.dcp_sharded for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    groups = make_kimi_cache_groups(
        mixed_attention_specs,
        mamba_specs,
        recurrent_group_size=len(mixed_attention_specs),
    )
    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups
