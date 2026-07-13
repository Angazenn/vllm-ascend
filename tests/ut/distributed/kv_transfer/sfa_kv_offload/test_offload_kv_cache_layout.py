"""Unit tests for the BF16 SFA decode-offload tuple layout."""

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.offload_kv_cache_layout import (
    OFFLOAD_INDEXER_K,
    OFFLOAD_MAIN_K,
    OFFLOAD_MAIN_V,
    OFFLOAD_RESIDENT_K,
    OFFLOAD_RESIDENT_V,
    OFFLOAD_TUPLE_LEN,
)


def test_tuple_len_matches_slot_indices():
    assert OFFLOAD_TUPLE_LEN == 5
    assert OFFLOAD_MAIN_K == 0
    assert OFFLOAD_MAIN_V == 1
    assert OFFLOAD_INDEXER_K == 2
    assert OFFLOAD_RESIDENT_K == 3
    assert OFFLOAD_RESIDENT_V == 4
