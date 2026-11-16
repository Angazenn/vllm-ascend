# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace

import torch
from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, MambaSpec


def test_window_and_draft_ownership_survive_layout_placement():
    spec = AttentionSpec(block_size=32, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    manager = SlidingWindowManager.layer_config(spec, sliding_window=128, extra_retained_tokens=7)
    draft = replace(manager, is_draft=True)
    padded = replace(draft.spec, page_size_padded=16384)
    group = KVCacheGroup.from_layers({"draft": replace(draft, spec=padded)}, is_eagle_group=True)

    assert group.manager_config == manager.manager_config
    assert group.is_eagle_group
    placed = group.kv_cache_specs["draft"]
    assert draft.is_draft
    assert placed.page_size_bytes == 16384
    assert group.sliding_window_reach == 134


def test_mamba_layout_padding_does_not_replace_checkpoint_policy():
    spec = MambaSpec(block_size=64, shapes=((2, 8), (4, 8)), dtypes=(torch.bfloat16, torch.float32))
    config = MambaManager.layer_config(
        spec,
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
    result = replace(config, spec=replace(config.spec, page_size_padded=512))

    assert result.manager_config == config.manager_config
    assert result.spec.page_size_bytes == 512
    assert result.manager_config.params.num_speculative_blocks == 3
    assert result.manager_config.params.mamba_cache_mode == "align"
