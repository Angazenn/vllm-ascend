from types import SimpleNamespace

import torch

import vllm_ascend.attention.sfa_v1 as sfa_v1
from vllm_ascend.attention.indexer import (
    AscendSFAOffloadIndexerMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def test_offload_indexer_metadata_uses_its_own_group_addressing():
    block_table = torch.arange(12, dtype=torch.int32).view(3, 4)
    slot_mapping = torch.arange(8, dtype=torch.int64)
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=5,
        num_actual_tokens=5,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )
    builder = AscendSFAOffloadIndexerMetadataBuilder.__new__(
        AscendSFAOffloadIndexerMetadataBuilder
    )
    metadata = builder.build(0, common)
    assert metadata.block_table_tensor.data_ptr() == block_table.data_ptr()
    assert metadata.block_table_tensor.shape == (2, 4)
    assert metadata.slot_mapping.data_ptr() == slot_mapping.data_ptr()
    assert metadata.slot_mapping.shape == (5,)


def test_direct_decode_composes_real_indexer_into_five_tuple():
    main_cache = tuple(torch.tensor([index]) for index in range(5))
    indexer_cache = (torch.tensor([99]),)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = True
    impl.use_sfa_decode_offload = True
    impl.use_offload = True
    impl.use_sparse_c8_indexer = False
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=indexer_cache)
    )

    composed = impl._compose_sfa_kv_cache(main_cache)
    assert composed is not None
    assert len(composed) == 5
    assert composed[0] is main_cache[0]
    assert composed[1] is main_cache[1]
    assert composed[2] is indexer_cache[0]
    assert composed[3] is main_cache[3]
    assert composed[4] is main_cache[4]


def test_direct_decode_skip_topk_keeps_main_tuple():
    main_cache = tuple(torch.tensor([index]) for index in range(5))
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = False
    impl.use_sfa_decode_offload = True
    assert impl._compose_sfa_kv_cache(main_cache) is main_cache


def test_direct_decode_mtp_uses_indexer_group_addressing(monkeypatch):
    block_table = torch.arange(12, dtype=torch.int32).view(3, 4)
    slot_mapping = torch.arange(8, dtype=torch.int32)
    layer_name = "model.layers.78.self_attn.attn"
    indexer_name = "model.layers.78.self_attn.indexer.k_cache"
    draft_metadata = SimpleNamespace(
        indexer_block_table_tensor=block_table,
        indexer_slot_mapping=slot_mapping,
    )
    monkeypatch.setattr(
        sfa_v1,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer_name: draft_metadata}),
    )

    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.has_indexer = True
    impl.use_sfa_decode_offload = True
    impl.layer_name = layer_name
    impl.indexer = SimpleNamespace(
        k_cache=SimpleNamespace(prefix=indexer_name)
    )

    metadata = impl._get_split_indexer_metadata()
    assert metadata.block_table_tensor.data_ptr() == block_table.data_ptr()
    assert metadata.slot_mapping.data_ptr() == slot_mapping.data_ptr()
