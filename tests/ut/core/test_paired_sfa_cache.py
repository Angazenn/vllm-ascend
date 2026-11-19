# SPDX-License-Identifier: Apache-2.0
"""Allocation descriptors and ownership for uncompressed paired SFA caches."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)
from vllm_ascend.patch.platform import patch_kv_cache_utils


def make_specs():
    specs = {}
    sparse = [*range(3, 92, 4), 92]
    for layer in range(93):
        name = f"model.layers.{layer}"
        if layer in sparse:
            specs[name + ".attn"] = AscendMLAAttentionSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=656,
                dtype=torch.int8,
                cache_sparse_sfa_c8=True,
            )
            specs[name + ".indexer.k_cache"] = AscendSFAIndexerCacheSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.int8,
                scale_dim=1,
                scale_dtype=torch.float32,
                cache_sparse_li_c8=True,
            )
        else:
            specs[name + ".linear_attn"] = MambaSpec(
                block_size=1536,
                shapes=((3, 4608), (12, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
                page_size_padded=1536 * 656,
            )
    return specs


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k31"),
            hf_text_config=SimpleNamespace(model_type="kimi_linear"),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
    )


def test_token_blocks_unchanged_and_components_share_ids():
    specs, config = make_specs(), make_config()
    groups = paired_sfa_cache_groups(config, specs)
    assert len(groups) == 4
    assert [len(g.layer_names) for g in groups] == [48, 23, 23, 23]
    assert all(isinstance(g.kv_cache_spec, MambaSpec) for g in groups[1:])
    assert {s.block_size for s in specs.values()} == {1536}
    assert {s.page_size_bytes for s in specs.values()} == {1007616, 202752}
    assert paired_sfa_pool_bytes(groups) == 29048832
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.num_blocks == 604
    backing_size = 29048832 * 604
    assert backing_size <= 17556701184
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {backing_size}
    assert len(plan.kv_cache_tensors) == 5
    assert len(groups[0].kv_cache_spec.kv_cache_specs) == 48
    main, *recurrent, indexer = plan.kv_cache_tensors
    assert len(main.layers) == len(indexer.layers) == 24
    assert main.offset == 0
    assert main.layer_stride == 1007616 * 604
    assert main.block_stride == 1007616
    for tensor, group in zip(recurrent, groups[1:]):
        assert tensor.layers == group.layer_names
        assert tensor.offset == main.offset
        assert tensor.layer_stride == main.layer_stride
        assert tensor.block_stride == main.block_stride
    assert indexer.offset == 24 * main.layer_stride
    assert indexer.layer_stride == 202752 * 604
    assert indexer.block_stride == 202752
    assert indexer.offset + len(indexer.layers) * indexer.layer_stride == backing_size


@pytest.mark.parametrize("interval", [None, 5])
def test_paired_config_propagates_prefix_cache_retention_interval(interval):
    config = make_config()
    config.cache_config.prefix_cache_retention_interval = interval
    groups = paired_sfa_cache_groups(config, make_specs())
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.prefix_cache_retention_interval == interval


@pytest.mark.parametrize(
    ("partition", "target_pair_counts"),
    [("25,25,24,19", [6, 6, 6, 6]), ("24,24,24,21", [6, 6, 6, 6]), ("3,90", [0, 24])],
)
def test_placement_uses_only_local_decoder_layers(monkeypatch, partition, target_pair_counts):
    from vllm.distributed.utils import get_pp_indices
    from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    specs = make_specs()
    config = make_config()
    groups = paired_sfa_cache_groups(config, specs)
    for rank, pair_count in enumerate(target_pair_counts):
        start, end = get_pp_indices(93, rank, len(target_pair_counts))
        local = {name: spec for name, spec in specs.items() if start <= int(name.split(".")[2]) < end}
        projected = _project_kv_cache_groups_to_worker(groups, local)
        plan = paired_sfa_cache_config(config, projected, 2**30)
        by_layer = {name: tensor for tensor in plan.kv_cache_tensors for name in tensor.layers}
        assert by_layer.keys() == local.keys()
        assert sum(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in local.values()) == pair_count
        for name, spec in local.items():
            tensor = by_layer[name]
            assert tensor.block_stride == spec.page_size_bytes
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                assert tensor.offset + len(tensor.layers) * tensor.layer_stride == tensor.size
            else:
                assert tensor.offset == 0
        if pair_count == 0:
            # Profiling groups the worker's local specs before global projection.
            profile_groups = paired_sfa_cache_groups(config, local)
            assert paired_sfa_pool_bytes(profile_groups) == sum(s.page_size_bytes for s in local.values())


def test_dspark_preserves_target_recurrent_budget_and_order(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    config = make_config()
    config.model_config.max_model_len = 8192
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1)
    target = make_specs()
    target_groups = paired_sfa_cache_groups(config, target)
    # Enough draft layers to change recurrent grouping if they were counted
    # as target overlay slots. Supply drafts first and targets in reverse order.
    draft = {
        f"model.layers.{layer}.attn": AscendMLAAttentionSpec(
            block_size=1536,
            num_kv_heads=1,
            head_size=576 if layer % 2 else 512,
            dtype=torch.bfloat16,
            is_draft=True,
            non_causal_multi_token_decode=True,
        )
        for layer in reversed(range(93, 108))
    }
    specs = {**draft, **dict(reversed(target.items()))}
    config.speculative_config = SimpleNamespace(method="dspark")
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    draft_names = [f"model.layers.{layer}.attn" for layer in range(93, 108)]
    assert [len(g.layer_names) for g in groups] == [63, 23, 23, 23]
    assert groups[0].layer_names == target_groups[0].layer_names + draft_names
    assert groups[0].is_eagle_group
    assert groups[1:] == target_groups[1:]
    assert groups[0].kv_cache_spec.kv_cache_specs == {**target_groups[0].kv_cache_spec.kv_cache_specs, **draft}
    draft_bytes = sum(spec.page_size_bytes for spec in draft.values())
    assert paired_sfa_pool_bytes(groups) == paired_sfa_pool_bytes(target_groups) + draft_bytes

    # A draft-only PP rank must use the original target geometry after the
    # upstream planner projects its group and converges the ranks' block counts.
    monkeypatch.setattr(
        current_platform,
        "get_kv_cache_config_builder_cls",
        lambda _: "vllm_ascend.core.kv_cache_config_builder.AscendKVCacheConfigBuilder",
    )
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.model_config.original_max_model_len = config.model_config.max_model_len
    config.cache_config.enable_extensible_kv_cache = False
    config.cache_config.mamba_cache_mode = "align"
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    target_budget = paired_sfa_pool_bytes(target_groups) * 40
    _, plan = get_kv_cache_configs(config, [target, draft], [target_budget, 32 * draft_bytes], groups)
    assert plan.num_blocks == 32
    offset = 0
    for tensor, name in zip((t for t in plan.kv_cache_tensors if t.layers), draft_names, strict=True):
        page = draft[name].page_size_bytes
        assert (tensor.layers, tensor.offset, tensor.block_stride, tensor.size) == (
            [name],
            offset,
            page,
            32 * draft_bytes,
        )
        offset += 32 * page


@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize("dcp_sharded", [False, True])
def test_dense_draft_retains_sharding_coverage_and_separate_storage(dcp_size, dcp_sharded):
    config = make_config()
    config.model_config.max_model_len = 32768
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=dcp_size)
    specs = make_specs()
    target_groups = paired_sfa_cache_groups(config, specs)
    name = "model.layers.93.attn"
    draft = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        is_draft=True,
        dcp_sharded=dcp_sharded,
    )
    groups = paired_sfa_cache_groups(config, {**specs, name: draft})
    draft_group = next(group for group in groups if name in group.layer_names)
    assert draft_group.kv_cache_spec.dcp_sharded == dcp_sharded
    assert not draft_group.is_eagle_group
    assert draft_group.kv_cache_spec.max_num_blocks_per_req(config, 32768) == draft.max_num_blocks_per_req(
        config, 32768
    )
    assert (draft_group is groups[0]) == dcp_sharded
    assert groups[1:4] == target_groups[1:]
    assert paired_sfa_pool_bytes(groups) == paired_sfa_pool_bytes(target_groups) + draft.page_size_bytes
    plan = paired_sfa_cache_config(config, groups, 2**30)
    draft_tensor = next(tensor for tensor in plan.kv_cache_tensors if tensor.layers == [name])
    assert draft_tensor.block_stride == draft.page_size_bytes
    assert draft_tensor.offset + draft_tensor.layer_stride == draft_tensor.size


def test_target_only_without_recurrent_layers_keeps_one_full_history_group():
    specs = {name: spec for name, spec in make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = paired_sfa_cache_groups(make_config(), specs)
    assert [len(g.layer_names) for g in groups] == [48]
    assert groups[0].kv_cache_spec.kv_cache_specs == specs
    assert paired_sfa_pool_bytes(groups) == 29048832


def test_mismatched_component_blocks_rejected():
    specs = make_specs()
    name = next(n for n in specs if n.endswith(".k_cache"))
    specs[name] = replace(specs[name], block_size=3072)
    with pytest.raises(ValueError, match="same local token block"):
        paired_sfa_cache_groups(make_config(), specs)


@pytest.mark.parametrize("model_type", ["kimi_k31", "kimi_k3", "deepseek_v32", "glm_moe_dsa"])
def test_paired_planner_selection_uses_outer_model_type(monkeypatch, model_type):
    config, specs = make_config(), make_specs()
    config.model_config.hf_config.model_type = model_type
    paired = Mock(return_value=[])
    upstream = Mock(return_value=[])
    monkeypatch.setattr(patch_kv_cache_utils, "paired_sfa_cache_groups", paired)
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_groups", upstream)
    monkeypatch.setattr(patch_kv_cache_utils, "get_qwen4_exp_kv_cache_groups", lambda *_: None)

    actual = patch_kv_cache_utils._get_ascend_kv_cache_groups(config, specs)

    # The same hybrid cache geometry must not opt unrelated models into K3.1.
    # K3.1's nested text config still reports kimi_linear.
    selected, other = (paired, upstream) if model_type == "kimi_k31" else (upstream, paired)
    assert actual is selected.return_value
    selected.assert_called_once_with(config, specs)
    other.assert_not_called()


def test_grouping_preserves_platform_aligned_specs():
    specs = make_specs()
    original = specs.copy()
    first = paired_sfa_cache_groups(make_config(), specs)
    second = paired_sfa_cache_groups(make_config(), specs)
    assert specs == original
    assert paired_sfa_pool_bytes(first) == paired_sfa_pool_bytes(second) == 29048832


def test_grouping_rejects_obsolete_state_padding():
    specs = {
        n: replace(s, page_size_padded=66502656) if isinstance(s, MambaSpec) else s for n, s in make_specs().items()
    }
    with pytest.raises(ValueError, match="recurrent pages must match"):
        paired_sfa_cache_groups(make_config(), specs)


def test_a3_int8_fp16_pages_preserve_tp16_recurrent_ownership():
    specs = {}
    for name, spec in make_specs().items():
        if isinstance(spec, MambaSpec):
            spec = replace(spec, shapes=((3, 2304), (6, 128, 128)), page_size_padded=768 * 656)
        elif isinstance(spec, AscendSFAIndexerCacheSpec):
            spec = replace(spec, scale_dtype=torch.float16)
        specs[name] = replace(spec, block_size=768)
    config = make_config()
    groups = paired_sfa_cache_groups(config, specs)
    # 24 packed-MLA pages (768*656 bytes) plus 24 INT8/FP16 indexer
    # pages (768*(128+2) bytes); recurrent checkpoints alias main slots.
    assert {spec.page_size_bytes for spec in specs.values()} == {503808, 99840}
    assert {spec.block_size for spec in specs.values()} == {768}
    assert paired_sfa_pool_bytes(groups) == 14487552
    plan = paired_sfa_cache_config(config, groups, 115900539)
    assert plan.num_blocks == 8
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {115900416}
    assert all(
        spec.sfa_dcp_replicated_indexer_size == 1
        for spec in specs.values()
        if isinstance(spec, AscendSFAIndexerCacheSpec)
    )
