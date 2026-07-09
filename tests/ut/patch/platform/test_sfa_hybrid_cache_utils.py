# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerAliasCacheSpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_groups,
    _get_sfa_hybrid_kv_cache_config_from_groups,
    _is_sfa_indexer_kv_cache_group,
)


def _make_config(num_layers: int = 8):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        model_config=SimpleNamespace(
            get_num_layers=lambda _parallel_config: num_layers,
        ),
        parallel_config=SimpleNamespace(),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )


def _make_specs(
    num_layers: int = 8,
    indexer_layer_ids: tuple[int, ...] = (0, 1),
):
    specs = {}
    for layer_id in range(num_layers):
        specs[f"model.layers.{layer_id}.self_attn"] = AscendMLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=6,
            dtype=torch.bfloat16,
        )
    for layer_id in indexer_layer_ids:
        specs[f"model.layers.{layer_id}.self_attn.indexer.k_cache"] = (
            AscendSFAIndexerAliasCacheSpec(
                block_size=6,
                num_kv_heads=1,
                head_size=4,
                dtype=torch.bfloat16,
            )
        )
    return specs


def test_sfa_hybrid_groups_use_real_indexer_cache_layers_only():
    vllm_config = _make_config()

    groups = _ascend_get_kv_cache_groups(vllm_config, _make_specs())

    assert len(groups) == 3
    assert all(_is_sfa_indexer_kv_cache_group(group) for group in groups[:2])
    assert not _is_sfa_indexer_kv_cache_group(groups[-1])
    assert [group.layer_names for group in groups[:2]] == [
        ["model.layers.0.self_attn.indexer.k_cache"],
        ["model.layers.1.self_attn.indexer.k_cache"],
    ]
    assert groups[-1].layer_names == [
        f"model.layers.{layer_id}.self_attn" for layer_id in range(8)
    ]


def test_sfa_hybrid_config_keeps_alias_layers_on_shared_physical_tensors():
    vllm_config = _make_config()
    groups = _ascend_get_kv_cache_groups(vllm_config, _make_specs())

    config = _get_sfa_hybrid_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=48 * 8 * 3,
    )

    assert config is not None
    assert len(config.kv_cache_groups) == 3
    assert len(config.kv_cache_tensors) == 4
    assert config.kv_cache_tensors[0].shared_by == [
        "model.layers.0.self_attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn",
        "model.layers.1.self_attn.indexer.k_cache",
    ]
    assert config.kv_cache_tensors[1].shared_by == [
        "model.layers.2.self_attn",
        "model.layers.3.self_attn",
    ]


def test_sfa_hybrid_groups_allow_sparse_real_indexer_layer_ids():
    vllm_config = _make_config()

    groups = _ascend_get_kv_cache_groups(
        vllm_config,
        _make_specs(indexer_layer_ids=(0, 2)),
    )

    assert len(groups) == 3
    assert [group.layer_names for group in groups[:2]] == [
        ["model.layers.0.self_attn.indexer.k_cache"],
        ["model.layers.2.self_attn.indexer.k_cache"],
    ]


def test_sfa_hybrid_groups_support_glm52_shared_indexer_pattern():
    indexer_layer_ids = (
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
    )
    vllm_config = _make_config(num_layers=80)

    groups = _ascend_get_kv_cache_groups(
        vllm_config,
        _make_specs(num_layers=80, indexer_layer_ids=indexer_layer_ids),
    )

    assert len(groups) == len(indexer_layer_ids) + 1
    assert [group.layer_names[0] for group in groups[:-1]] == [
        f"model.layers.{layer_id}.self_attn.indexer.k_cache"
        for layer_id in indexer_layer_ids
    ]
    assert groups[-1].layer_names == [
        f"model.layers.{layer_id}.self_attn" for layer_id in range(80)
    ]

    config = _get_sfa_hybrid_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=48 * 80 * 3,
    )

    assert config is not None
    assert len(config.kv_cache_groups) == len(indexer_layer_ids) + 1
    assert len(config.kv_cache_tensors) == 4
    assert "model.layers.6.self_attn.indexer.k_cache" in (
        config.kv_cache_tensors[0].shared_by
    )
    assert "model.layers.22.self_attn.indexer.k_cache" in (
        config.kv_cache_tensors[1].shared_by
    )
    assert "model.layers.74.self_attn.indexer.k_cache" in (
        config.kv_cache_tensors[3].shared_by
    )
