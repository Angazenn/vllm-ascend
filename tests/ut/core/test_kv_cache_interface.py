import torch
from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

from vllm_ascend.core.kv_cache_interface import (
    AscendSFAOffloadIndexerCacheSpec,
    AscendSFALayerwiseIndexerCacheSpec,
    is_direct_sfa_kv_offload,
    make_offload_indexer_mla_spec,
    make_offload_main_mla_spec,
    make_sfa_offload_indexer_spec,
    offload_indexer_kernel_block_size,
    offload_indexer_pad_dim,
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
INDEXER_PAD_DIM = offload_indexer_pad_dim(
    INDEX_HEAD_DIM, QK_ROPE_HEAD_DIM, KV_LORA_RANK
)
MLA_BLOCK_SIZE = 128


def test_layerwise_bf16_indexer_page_unifies_with_main_mla():
    spec = AscendSFALayerwiseIndexerCacheSpec(
        block_size=MLA_BLOCK_SIZE * 4,
        num_kv_heads=1,
        head_size=INDEX_HEAD_DIM,
        indexer_pad_dim=INDEXER_PAD_DIM,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    main_page_size = (
        MLA_BLOCK_SIZE
        * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        * torch.bfloat16.itemsize
    )
    assert spec.page_size_bytes == main_page_size


def test_legacy_decode_offload_specs_are_bf16_only():
    main = make_offload_main_mla_spec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
    )
    indexer = make_offload_indexer_mla_spec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=INDEX_HEAD_DIM + INDEXER_PAD_DIM,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        index_head_dim=INDEX_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        sparse_head_dim=(
            KV_LORA_RANK,
            QK_ROPE_HEAD_DIM,
            INDEX_HEAD_DIM,
        ),
    )
    assert main.page_size_bytes == (
        MLA_BLOCK_SIZE
        * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        * torch.bfloat16.itemsize
    )
    assert indexer.cache_sparse_c8 is False
    assert indexer.scale_dim == 0
    assert indexer.page_size_bytes == (
        MLA_BLOCK_SIZE
        * (INDEX_HEAD_DIM + INDEXER_PAD_DIM)
        * torch.bfloat16.itemsize
    )
    assert offload_indexer_kernel_block_size(
        MLA_BLOCK_SIZE, KV_LORA_RANK, INDEX_HEAD_DIM
    ) == MLA_BLOCK_SIZE * 4


def test_direct_decode_indexer_page_unifies_with_main_mla():
    main = make_offload_main_mla_spec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
    )
    indexer = make_sfa_offload_indexer_spec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        index_head_dim=INDEX_HEAD_DIM,
        indexer_pad_dim=INDEXER_PAD_DIM,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    assert isinstance(indexer, AscendSFAOffloadIndexerCacheSpec)
    unified = unify_kv_cache_spec_page_size(
        {"indexer": indexer, "main": main}
    )
    assert unified["indexer"].block_size == MLA_BLOCK_SIZE * 4
    assert unified["main"].block_size == MLA_BLOCK_SIZE
    assert (
        unified["indexer"].page_size_bytes
        == unified["main"].page_size_bytes
    )


def test_direct_decode_offload_requires_exact_layerwise_connector():
    def config(connector: str, use_layerwise: bool = True):
        return SimpleNamespace(
            additional_config={"use_offload": True},
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_extra_config={
                    "use_layerwise": use_layerwise
                },
            ),
        )

    assert is_direct_sfa_kv_offload(config("SFAKVOffloadConnector"))
    assert not is_direct_sfa_kv_offload(config("AscendStoreConnector"))
    assert not is_direct_sfa_kv_offload(config("MultiConnector"))
    assert not is_direct_sfa_kv_offload(
        config("SFAKVOffloadConnector", use_layerwise=False)
    )
