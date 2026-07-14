"""Unit tests for the SFA sparse KV offload tuple-slot layout helpers.

Covers the six-tuple (LIC8) contract and the owner-shared seven-tuple BF16
contract, whose last two entries are layer-private tail K/V caches.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import (
    make_offload_main_mla_spec,
    make_sfa_offload_indexer_spec,
)

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.offload_kv_cache_layout import (
    OFFLOAD_C8_TUPLE_LEN,
    OFFLOAD_INDEXER_K,
    OFFLOAD_INDEXER_S,
    OFFLOAD_LEGACY_TUPLE_LEN,
    OFFLOAD_MAIN_K,
    OFFLOAD_MAIN_V,
    OFFLOAD_RESIDENT_K,
    OFFLOAD_RESIDENT_V,
    OFFLOAD_TAIL_K,
    OFFLOAD_TAIL_V,
    OFFLOAD_TUPLE_LEN,
    build_sfa_offload_shared_cache_plan,
    is_offload_c8_kv_cache,
)


def test_tuple_len_constants_match_slot_indices():
    assert OFFLOAD_TUPLE_LEN == 7
    assert OFFLOAD_C8_TUPLE_LEN == 6
    assert OFFLOAD_LEGACY_TUPLE_LEN == 5
    assert OFFLOAD_MAIN_K == 0
    assert OFFLOAD_MAIN_V == 1
    assert OFFLOAD_INDEXER_K == 2
    assert OFFLOAD_RESIDENT_K == 3
    assert OFFLOAD_RESIDENT_V == 4
    assert OFFLOAD_INDEXER_S == 5
    assert OFFLOAD_TAIL_K == 5
    assert OFFLOAD_TAIL_V == 6


def test_is_offload_c8_kv_cache_detects_six_tuple():
    six = tuple(torch.zeros(1) for _ in range(6))
    seven = tuple(torch.zeros(1) for _ in range(7))
    assert is_offload_c8_kv_cache(six) is True
    assert is_offload_c8_kv_cache(seven) is False


def test_is_offload_c8_kv_cache_rejects_other_lengths():
    for n in (0, 1, 2, 3, 4, 5, 8):
        tup = tuple(torch.zeros(1) for _ in range(n))
        assert is_offload_c8_kv_cache(tup) is False


def _layer(layer_id: int, role: str) -> str:
    return f"model.layers.{layer_id}.self_attn.{role}"


def _make_plan(
    main_ids: list[int],
    owner_ids: list[int],
    *,
    num_hidden_layers: int = 75,
    resident_capacity: int | None = 32,
):
    main_spec = make_offload_main_mla_spec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = make_sfa_offload_indexer_spec(
        main_block_size=128,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="SFAKVOffloadConnector"
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_hidden_layers=num_hidden_layers
            )
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=64,
        ),
        speculative_config=None,
        additional_config={
            "lru_resident_cache_config": {
                "enabled": True,
                "buffer_size": 48,
                "topk": 24,
            }
        },
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=indexer_spec,
            layer_names=[
                _layer(layer_id, "indexer.k_cache")
                for layer_id in owner_ids
            ],
        ),
        SimpleNamespace(
            kv_cache_spec=main_spec,
            layer_names=[
                _layer(layer_id, "attn") for layer_id in main_ids
            ],
        ),
    ]
    return build_sfa_offload_shared_cache_plan(
        config,
        groups,
        resident_capacity=resident_capacity,
    )


def test_glm52_owner_segments_and_page_compatibility():
    plan = _make_plan(
        main_ids=list(range(11)),
        owner_ids=[0, 1, 2, 6, 10],
    )
    assert plan is not None
    assert plan.indexer_group_id == 0
    assert plan.main_group_id == 1
    assert [
        tuple(int(name.split(".layers.")[1].split(".")[0])
              for name in pool.main_layer_names)
        for pool in plan.pools
    ] == [(0,), (1,), (2, 3, 4, 5), (6, 7, 8, 9), (10,)]
    assert plan.main_page_bytes == 128 * (512 + 64) * 2


def test_glm51_every_main_layer_owns_a_pool():
    plan = _make_plan(main_ids=[0, 1, 2, 3], owner_ids=[0, 1, 2, 3])
    assert plan is not None
    assert plan.num_physical_pools == 4
    assert all(len(pool.main_layer_names) == 1 for pool in plan.pools)


def test_plan_reads_resident_capacity_from_vllm_config():
    plan = _make_plan(
        main_ids=[0, 1],
        owner_ids=[0, 1],
        resident_capacity=None,
    )
    assert plan is not None
    # max_num_topk_rows=4 and each main token stores 576 BF16 elements.
    expected_resident_bytes = 2 * 4 * 48 * 576 * 2
    expected_tail_bytes = 2 * 4 * 2 * plan.main_page_bytes
    assert plan.fixed_hbm_bytes == (
        expected_resident_bytes + expected_tail_bytes
    )


def test_mtp_main_requires_real_indexer_owner():
    with pytest.raises(ValueError, match="MTP main layer"):
        _make_plan(
            main_ids=[0, 1, 2, 3],
            owner_ids=[0, 2],
            num_hidden_layers=3,
        )
