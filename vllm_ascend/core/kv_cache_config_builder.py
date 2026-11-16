# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter, defaultdict
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder, _approximate_gcd
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroup, KVCacheLayerConfig, KVCacheTensor

from vllm_ascend.models.deepseek_v41.cache_config import (
    group_layer_configs,
    is_deepseek_v41_cache,
    place_deepseek_v41_cache,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


def _is_deepseek_v4(specs):
    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Apply Ascend layouts with native block-table groups and manager policy."""

    def get_kv_cache_groups(
        self, vllm_config: VllmConfig, layer_configs: dict[str, KVCacheLayerConfig]
    ) -> list[KVCacheGroup]:
        specs = [config.spec for config in layer_configs.values()]
        if is_deepseek_v41_cache(specs):
            return group_layer_configs(layer_configs)
        if _is_deepseek_v4(specs):
            return _group_deepseek_v4_layers(layer_configs)
        if getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31":
            return paired_sfa_cache_groups(vllm_config, layer_configs)
        if any(is_glm5_next_cache_spec(spec) for spec in specs):
            return get_glm5_next_kv_cache_groups(vllm_config, layer_configs)
        groups = get_qwen4_exp_kv_cache_groups(vllm_config, layer_configs, super().get_kv_cache_groups)
        if groups is None:
            groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(layer_configs)
        if groups is None:
            return super().get_kv_cache_groups(vllm_config, layer_configs)
        return groups

    def get_kv_cache_config_from_groups(
        self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroup], num_blocks: int
    ) -> KVCacheConfig:
        specs = tuple(spec for group in kv_cache_groups for spec in group.distinct_specs)
        if is_deepseek_v41_cache(specs):
            return place_deepseek_v41_cache(vllm_config, kv_cache_groups, num_blocks)
        if not _is_deepseek_v4(specs):
            if getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31":
                return paired_sfa_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
            if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
                return get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
            config = get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
            if config is not None:
                return config
            return super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)

        if tq_cache.uses_turboquant_groups(kv_cache_groups):
            tensors = tq_cache.cache_tensors(kv_cache_groups, num_blocks)
        else:
            tensors = _place_deepseek_v4_cache(kv_cache_groups, num_blocks)
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )

    def _get_pool_bytes_per_block(self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroup]) -> int | None:
        if getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31":
            return paired_sfa_pool_bytes(kv_cache_groups)
        if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
            return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
        return get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)


def _group_deepseek_v4_layers(layer_configs: dict[str, KVCacheLayerConfig]) -> list[KVCacheGroup]:
    full = defaultdict(dict)
    local = defaultdict(dict)
    for name, config in layer_configs.items():
        if config.manager_cls is SlidingWindowManager:
            local[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
        else:
            full[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
    full_groups = [layers for _, layers in sorted(full.items(), key=lambda item: item[0][0])]
    grouped_layers = [*full_groups, *local.values()]
    if any(
        getattr(config.spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE for config in layer_configs.values()
    ):
        return tq_cache.group_layers(grouped_layers)

    if not full_groups:
        return [KVCacheGroup.from_layers(group) for group in grouped_layers]
    page_sizes = {config.spec.page_size_bytes for config in full_groups[0].values()}
    counts = [
        max(Counter(config.spec.page_size_bytes for config in group.values()).values()) for group in grouped_layers
    ]
    num_tuples = _approximate_gcd(counts, lower_bound=counts[0])
    groups = [KVCacheGroup.from_layers(group) for group in full_groups]
    for layers in local.values():
        buckets = defaultdict(list)
        for name, config in layers.items():
            current_size = config.spec.page_size_bytes
            page_size = min(size for size in page_sizes if size >= current_size)
            if page_size != current_size:
                config = replace(config, spec=replace(config.spec, page_size_padded=page_size))
            buckets[page_size].append((name, config))
        assert len({len(bucket) for bucket in buckets.values()}) == 1
        tuples = list(zip(*buckets.values(), strict=True))
        num_groups = cdiv(len(tuples), num_tuples)
        for i in range(num_groups):
            groups.append(KVCacheGroup.from_layers(dict(pair for slot in tuples[i::num_groups] for pair in slot)))
    return groups


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroup],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream so every layer's page_size matches one of the
    full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    page_sizes = sorted({spec.page_size_bytes for spec in kv_cache_groups[0].distinct_specs})

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        specs = group.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            padded = getattr(specs[name], "page_size_padded", None)
            page_size = padded if padded is not None else specs[name].page_size_bytes
            if "mtp" not in name:
                b[page_size].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = page_size
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _place_deepseek_v4_cache(kv_cache_groups: list[KVCacheGroup], num_blocks: int) -> list[KVCacheTensor]:
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = sum(page_sizes)
    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * num_tuple_slots

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in bucketed:
        for page_size in page_sizes:
            layer_names = group_buckets.get(page_size)
            if not layer_names:
                continue

            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=list(layer_names),
                    offset=page_offsets[page_size],
                    layer_stride=tuple_stride,
                    block_stride=page_size,
                )
            )

    # MTP layers receive trailing tuple slots. Unused page buckets remain
    # padding so num_blocks and memory accounting retain the existing contract.
    normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
    for index, layer_name in enumerate(mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[mtp_page_size],
                layer_stride=0,
                block_stride=mtp_page_size,
            )
        )

    return tensors
