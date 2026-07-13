from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import (
    AscendSFAOffloadIndexerCacheSpec,
    OffloadMLAAttentionSpec,
    make_offload_main_mla_spec,
    make_sfa_offload_indexer_spec,
    offload_indexer_pad_dim,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_glm_sfa_decode_offload_kv_cache_groups,
)


def _config():
    return SimpleNamespace(
        additional_config={"use_offload": True},
        kv_transfer_config=SimpleNamespace(
            kv_connector="SFAKVOffloadConnector",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def _main_name(layer_id: int) -> str:
    return f"model.layers.{layer_id}.self_attn.attn"


def _indexer_name(layer_id: int) -> str:
    return f"model.layers.{layer_id}.self_attn.indexer.k_cache"


def _specs(num_layers: int, indexer_layer_ids: list[int]):
    main_spec = make_offload_main_mla_spec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = make_sfa_offload_indexer_spec(
        block_size=128,
        num_kv_heads=1,
        index_head_dim=128,
        indexer_pad_dim=offload_indexer_pad_dim(128, 64, 512),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    specs = {
        _main_name(layer_id): main_spec
        for layer_id in range(num_layers)
    }
    specs.update(
        {
            _indexer_name(layer_id): indexer_spec
            for layer_id in indexer_layer_ids
        }
    )
    return specs


@pytest.mark.parametrize(
    "indexer_layer_ids",
    [
        list(range(78)),
        [0, 1, 2, *range(6, 78, 4)],
    ],
)
def test_decode_groups_generalize_real_indexer_ownership(
    indexer_layer_ids: list[int],
):
    groups = _get_glm_sfa_decode_offload_kv_cache_groups(
        _config(), _specs(78, indexer_layer_ids)
    )
    assert groups is not None
    assert len(groups) == 2
    assert isinstance(
        groups[0].kv_cache_spec, AscendSFAOffloadIndexerCacheSpec
    )
    assert isinstance(groups[1].kv_cache_spec, OffloadMLAAttentionSpec)
    assert groups[0].layer_names == [
        _indexer_name(layer_id) for layer_id in indexer_layer_ids
    ]
    assert groups[1].layer_names == [
        _main_name(layer_id) for layer_id in range(78)
    ]
    assert groups[0].kv_cache_spec.block_size == 512
    assert groups[1].kv_cache_spec.block_size == 128


def test_decode_groups_reject_non_main_indexer_owner():
    with pytest.raises(ValueError, match="subset of main"):
        _get_glm_sfa_decode_offload_kv_cache_groups(
            _config(), _specs(2, [0, 2])
        )
