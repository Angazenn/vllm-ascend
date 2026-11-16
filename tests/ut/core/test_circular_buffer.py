# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroup,
    MambaSpec,
)


def ring_spec():
    return AttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1024,
        head_size_v=0,
        dtype=torch.float32,
    )


def test_ring_lifetime_reuse_and_external_tokens():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=32)
    manager = CircularBufferManager(
        KVCacheGroup.from_layers({"ring": CircularBufferManager.layer_config(ring_spec())}),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=128,
        needs_kv_cache_zeroing=True,
    )
    free = pool.get_num_free_blocks()
    assert manager.get_num_blocks_to_allocate("a", 129, [], 0, 0, 129) == 1
    a = manager.allocate_new_blocks("a", 129, 129)
    b = manager.allocate_new_blocks("b", 1, 1)
    assert len(a) == len(b) == 1 and a[0].block_id != b[0].block_id
    assert not a[0].is_null and pool.get_num_free_blocks() == free - 2
    for tokens in (130, 1024, 65536):
        assert manager.get_num_blocks_to_allocate("a", tokens, [], tokens - 1, 0, tokens) == 0
        assert manager.allocate_new_blocks("a", tokens, tokens) == []
        manager.allocate_external_computed_blocks("a", 0, tokens)
        manager.remove_skipped_blocks("a", tokens)
        manager.cache_blocks(
            SimpleNamespace(request_id="a"),
            tokens,
            **{"replay_boundaries": [tokens - 1]},
        )
        assert manager.req_to_blocks["a"] == a
    assert manager.take_new_block_ids() == []
    # vLLM main moved the prefix-block query up to ``KVCacheManager`` and
    # dropped it from single-type managers, so check the ring's own view.
    assert manager.get_num_skipped_tokens(65536) == 0
    manager.free("a")  # Finish or preempt, then resume under a new allocation.
    manager.free("b")
    assert pool.get_num_free_blocks() == free
    manager.allocate_external_computed_blocks("a", 0, 65536)
    assert len(manager.req_to_blocks["a"]) == 1
    manager.free("a")
    assert pool.get_num_free_blocks() == free


def test_scratch_groups_do_not_reduce_prefix_hits_or_truncation():
    full = AttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.float32)
    scratch = ring_spec()
    groups = [
        KVCacheGroup.from_layers({"kv": FullAttentionManager.layer_config(full)}),
        KVCacheGroup.from_layers({"state": CircularBufferManager.layer_config(scratch)}),
    ]
    assert [group.prefix_cacheable for group in groups] == [True, False]
    assert scratch.page_size_bytes == 131072
    host = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_size=128), SimpleNamespace(block_size=32)]
        ),
        create_kv_cache_blocks=lambda blocks: blocks,
    )
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [])), 128) == ([1], [])

    mamba = MambaSpec(block_size=128, shapes=((1,),), dtypes=(torch.float32,))
    host.kv_cache_config.kv_cache_groups = [
        groups[0],
        KVCacheGroup.from_layers({"mamba": MambaManager.layer_config(mamba, mamba_cache_mode="align")}),
        groups[1],
    ]
    host.coordinator.single_type_managers = [
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=32),
    ]
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [], [])), 128) == ([1], [], [])
