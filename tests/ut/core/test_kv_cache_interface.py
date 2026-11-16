# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroup,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    resolve_bounded_replay,
    supports_bounded_replay,
    supports_component_major_mla_pd,
)


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("replicas", [1, 2, 8])
@pytest.mark.parametrize("block_size", [128, 768])
@pytest.mark.parametrize(
    "dtype,head_size,scale_dim,scale_dtype",
    [
        (torch.bfloat16, 128, 0, torch.int8),
        (torch.int8, 128, 1, torch.float16),
        (torch.uint8, 64, 4, torch.float8_e8m0fnu),
    ],
)
def test_indexer_layout_strides_cover_replicated_keys_and_scales(
    replicas, block_size, dtype, head_size, scale_dim, scale_dtype
):
    spec = AscendSFAIndexerCacheSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
        sfa_dcp_replicated_indexer_size=replicas,
    )
    num_blocks = 3
    layer_stride, block_stride, *_ = compute_layout_strides(spec, num_blocks, 2, KVCacheLayout.LBNHC)
    assert block_stride == spec.page_size_bytes
    assert layer_stride == num_blocks * spec.page_size_bytes
    assert spec.unpadded_page_size_bytes == spec.page_size_bytes


@pytest.mark.parametrize("dcp_sharded,expected_blocks", [(True, 3), (False, 9)])
def test_ascend_mla_memory_uses_spec_dcp_layout(dcp_sharded, expected_blocks):
    spec = replace(_mla_spec(), dcp_sharded=dcp_sharded)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=129),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
    )
    assert spec.max_memory_usage_bytes(config) == expected_blocks * spec.page_size_bytes


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_get_storage_block_size_for_plain_mla_subclass():
    # vLLM #57169 removed `MLAAttentionSpec.storage_block_size` in both
    # supported lanes. A plain MLA subclass (HiddenStateCacheSpec) must
    # resolve to block_size.
    spec = HiddenStateCacheSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    expected = spec.block_size
    assert get_storage_block_size(spec) == expected


@pytest.mark.parametrize("ascend", [False, True])
@pytest.mark.parametrize("zeroing", [False, True])
def test_circular_group_preserves_spec_and_reports_recycled_pages(ascend, zeroing):
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

    spec_type = AscendCircularBufferSpec if ascend else AttentionSpec
    spec = spec_type(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    manager_type = AscendCircularBufferManager if ascend else CircularBufferManager
    group = KVCacheGroup.from_layers({"ring": manager_type.layer_config(spec)})
    pool = BlockPool(4, False, 32)
    # Reuse physical IDs released by another cache group.
    old_blocks = pool.get_new_blocks(pool.get_num_free_blocks())
    old_ids = {block.block_id for block in old_blocks}
    pool.free_blocks(old_blocks)
    manager = manager_type(group, pool, False, 0, scheduler_block_size=32, needs_kv_cache_zeroing=zeroing)
    blocks = manager.allocate_new_blocks("request", 1, 1)
    assert len(blocks) == 1 and blocks[0].block_id in old_ids
    assert manager.new_block_ids == ([blocks[0].block_id] if ascend and zeroing else [])
    assert manager.allocate_new_blocks("request", 2, 2) == []
    assert not group.prefix_cacheable and not group.uses_slot_mapping
    manager.free("request")
    assert pool.get_num_free_blocks() == len(old_ids)


def test_component_major_mla_pd_gate():
    assert supports_component_major_mla_pd(SimpleNamespace(kv_transfer_config=None))

    for connector in ("MooncakeConnectorV2", "MooncakePullConnector"):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_module_path=None,
            )
        )
        assert supports_component_major_mla_pd(config)

    for connector in ("MooncakeConnectorV1", "AscendStoreConnector", "MultiConnector"):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_module_path=None,
            )
        )
        assert not supports_component_major_mla_pd(config)

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
            kv_connector_module_path="custom.connector",
        )
    )
    assert not supports_component_major_mla_pd(config)


# --- the fields a hand-rebuilt spec has to carry -----------------------------


def _vllm_config(
    *, v2_runner: bool = False, pcp_size: int = 1, dp_size: int = 1, architectures=("DeepseekV41ForCausalLM",)
):
    return SimpleNamespace(
        use_v2_model_runner=v2_runner,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size, data_parallel_size=dp_size),
        model_config=SimpleNamespace(architectures=list(architectures)),
    )


def _cache_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"swa_bounded_replay": True, "enable_prefix_caching": True, **overrides})


def test_only_the_v4_1_target_may_replay():
    """The feature is a V4.1 one: upstream wires it into the V4.1 tree and that
    tree alone, picking it by architecture name. A V4 model has to come out off
    however the switch is set -- Ascend serves both from one model class."""
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]))
    assert supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]))
    assert supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41ForConditionalGeneration"]))


def test_the_v4_1_draft_is_not_recognised():
    """The DSpark draft is V4.1 too, but its SWA cache is a different class --
    ``DeepseekV41DSparkSWACache`` -- and that class does not carry the field yet,
    so a draft name here would open a gate onto nothing: the switch reads as on
    and no block replays. Left out until that class is wired as well."""
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41DSparkModel"]))
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DSparkV41DraftModel"]))
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41DSparkModel", "DSparkV41DraftModel"]))


def test_a_model_without_architectures_may_not_replay():
    """Absent, empty or a non-list: none of them is the V4.1 name, and the
    feature must not be inherited by accident."""
    assert not supports_bounded_replay(SimpleNamespace())
    assert not supports_bounded_replay(SimpleNamespace(architectures=None))
    assert not supports_bounded_replay(SimpleNamespace(architectures=[]))
    assert not supports_bounded_replay(None)


def test_a_v4_layer_never_replays_however_the_switch_is_set():
    with patch("vllm.logger.logger.warning_once") as warning:
        assert not resolve_bounded_replay(_vllm_config(architectures=["DeepseekV4ForCausalLM"]), _cache_config())

    # Silent, like upstream: the V4 tree does not read the switch at all, so
    # there is nothing to explain to the user.
    warning.assert_not_called()


def test_a_baseline_without_the_switch_replays_nothing():
    """The switch arrives with #56227. Where it does not exist the feature is
    unreachable rather than off -- there is no flag to have turned on."""
    assert not resolve_bounded_replay(_vllm_config(), SimpleNamespace())
    assert not resolve_bounded_replay(_vllm_config(), None)
    assert not resolve_bounded_replay(_vllm_config(), _cache_config(swa_bounded_replay=False))


def test_the_upstream_switch_is_all_that_is_left():
    """A V4.1 layer on the V1 runner replays without being asked twice: the
    scheduler conditions above have already had their say, and there is no
    capability left for this side to withhold."""
    with patch("vllm.logger.logger.warning_once") as warning:
        assert resolve_bounded_replay(_vllm_config(), _cache_config())

    warning.assert_not_called()


def test_the_v2_runner_keeps_supported_replay_enabled():
    """Ascend V2 pads replayed writes in its model-state attention setup."""
    assert resolve_bounded_replay(_vllm_config(v2_runner=True), _cache_config())


def test_the_v2_runner_does_not_warn():
    """V2 has its own slot-padding implementation for supported replay."""
    with patch("vllm.logger.logger.warning_once") as warning:
        resolve_bounded_replay(_vllm_config(v2_runner=True), _cache_config())

    warning.assert_not_called()


def test_prefill_context_parallelism_with_prefix_caching_keeps_replay_on():
    assert resolve_bounded_replay(_vllm_config(v2_runner=True, pcp_size=2), _cache_config())


def test_pcp_without_prefix_caching_keeps_replay_on():
    with patch("vllm.logger.logger.warning_once") as warning:
        assert resolve_bounded_replay(
            _vllm_config(v2_runner=True, pcp_size=4), _cache_config(enable_prefix_caching=False)
        )
    warning.assert_not_called()


def test_pcp_with_dp_keeps_replay_on():
    with patch("vllm.logger.logger.warning_once") as warning:
        assert resolve_bounded_replay(_vllm_config(v2_runner=True, pcp_size=4, dp_size=2), _cache_config())

    warning.assert_not_called()


def test_a_layer_that_may_replay_warns_about_nothing():
    """A layer that is allowed to replay is the quiet case: whatever is said here
    is said about a run that will not replay. Only the conditions that turn the
    switch off speak."""
    with patch("vllm.logger.logger.warning_once") as warning:
        resolve_bounded_replay(_vllm_config(), _cache_config())

    warning.assert_not_called()
