# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace

import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups


def _make_kimi_k3_dspark_kv_cache_specs(
    *,
    block_size: int = 384,
    page_size: int = 488448,
    target_layer_count: int = 24,
    draft_layer_count: int = 5,
    mamba_layer_count: int = 69,
) -> dict:
    target_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
        dcp_sharded=True,
    )
    draft_attention_spec = target_spec
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((10, 2304), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
    )
    specs = {
        f"language_model.model.layers.{layer_idx}.self_attn.attn": FullAttentionManager.layer_config(target_spec)
        for layer_idx in range(target_layer_count)
    }
    specs.update(
        {
            f"model.layers.{layer_idx}.self_attn.attn": replace(
                FullAttentionManager.layer_config(draft_attention_spec), is_draft=True
            )
            for layer_idx in range(93, 93 + draft_layer_count)
        }
    )
    specs.update(
        {
            f"language_model.model.layers.{layer_idx}.self_attn": MambaManager.layer_config(
                mamba_spec, mamba_cache_mode="align", num_speculative_blocks=7
            )
            for layer_idx in range(mamba_layer_count)
        }
    )
    return specs


def test_kimi_k3_mixed_attention_still_requires_same_block_size() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is not None
    name = "model.layers.93.self_attn.attn"
    specs[name] = replace(specs[name], spec=replace(specs[name].spec, block_size=192))
    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_kimi_k3_gqa_mixed_grouping_falls_back_on_unrecognized_layer() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    specs["unrecognized.layer"] = next(iter(specs.values()))

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_kimi_k3_dspark_group_count_is_derived_from_layer_ratio() -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        _make_kimi_k3_dspark_kv_cache_specs(
            target_layer_count=20,
            draft_layer_count=4,
            mamba_layer_count=70,
        )
    )

    assert groups is not None
    assert [len(group.layer_names) for group in groups] == [24, 24, 23, 23]


def test_kimi_k3_dspark_mixed_grouping_falls_back_on_unaligned_pages() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    draft_layer = "model.layers.93.self_attn.attn"
    specs[draft_layer] = replace(specs[draft_layer], spec=replace(specs[draft_layer].spec, page_size_padded=976896))

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None
