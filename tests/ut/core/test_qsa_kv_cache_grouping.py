from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroup,
    KVCacheLayerConfig,
)

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder


def _ordinary_specs() -> dict[str, KVCacheLayerConfig]:
    ordinary = AttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": FullAttentionManager.layer_config(ordinary)}


def test_non_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _ordinary_specs()
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")))
    expected = [Mock(spec=KVCacheGroup)]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(
        DefaultKVCacheConfigBuilder,
        "get_kv_cache_groups",
        upstream_grouping,
    )

    actual = AscendKVCacheConfigBuilder().get_kv_cache_groups(
        vllm_config,
        specs,
    )

    assert actual is expected
    upstream_grouping.assert_called_once_with(vllm_config, specs)


def test_non_qsa_config_uses_upstream_vllm_planner(monkeypatch) -> None:
    groups = [KVCacheGroup.from_layers(_ordinary_specs())]
    expected = SimpleNamespace()
    upstream_planner = Mock(return_value=expected)
    monkeypatch.setattr(
        DefaultKVCacheConfigBuilder,
        "get_kv_cache_config_from_groups",
        upstream_planner,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        kv_transfer_config=None,
    )

    actual = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        num_blocks=123,
    )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None
