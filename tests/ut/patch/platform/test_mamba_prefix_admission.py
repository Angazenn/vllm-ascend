# SPDX-License-Identifier: Apache-2.0
"""Admission counts must distinguish locally computed and transferred state."""

from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup, MambaSpec


def make_coordinator():
    specs = [
        AttentionSpec(block_size=b, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, dcp_sharded=True)
        for b in (251904, 50688)
    ] + [MambaSpec(block_size=1536, shapes=((1,),), dtypes=(torch.float32,)) for _ in range(3)]
    config = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroup.from_layers(
                {
                    str(i): MambaManager.layer_config(spec, mamba_cache_mode="align")
                    if isinstance(spec, MambaSpec)
                    else FullAttentionManager.layer_config(spec)
                }
            )
            for i, spec in enumerate(specs)
        ],
    )
    return HybridKVCacheCoordinator(
        config,
        max_model_len=16384,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=768,
        scheduler_block_size=8312832,
        max_in_flight_tokens=4096,
    )


def test_partial_boundary_decode_fits_available_blocks():
    """Regression for a real 768-token request preempting with six free pages."""
    coordinator = make_coordinator()
    request = SimpleNamespace(
        request_id="boundary",
        block_hashes=[b"x" * 32],
        num_prompt_tokens=768,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(request.request_id, 768, 768)
    coordinator.cache_blocks(request, 768)
    coordinator.new_step_starts()
    available = coordinator.block_pool.get_num_free_blocks()
    needed = coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=769,
        new_computed_blocks=tuple(() for _ in range(5)),
        num_encoder_tokens=0,
        total_computed_tokens=768,
        num_local_computed_tokens=768,
        num_tokens_main_model=769,
    )
    assert available == 6
    assert needed <= available
    coordinator.allocate_new_blocks(request.request_id, 769, 769)
    assert coordinator.block_pool.get_num_free_blocks() >= 0


def test_dcp8_prefix_reuses_checkpoint_and_copies_partial_attention_tail():
    specs = [AttentionSpec(block_size=1536, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, dcp_sharded=True)] + [
        MambaSpec(block_size=1536, shapes=((1,),), dtypes=(torch.float32,)) for _ in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroup.from_layers(
                {
                    str(i): MambaManager.layer_config(spec, mamba_cache_mode="align")
                    if isinstance(spec, MambaSpec)
                    else FullAttentionManager.layer_config(spec)
                }
            )
            for i, spec in enumerate(specs)
        ],
    )
    coordinator = HybridKVCacheCoordinator(
        config,
        max_model_len=262144,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=8,
        pcp_world_size=1,
        hash_block_size=1536,
        scheduler_block_size=12288,
        max_in_flight_tokens=4096,
    )
    assert coordinator.enable_partial_hash_hits
    assert coordinator._cache_hit_alignment_tokens == 1536
    assert [m.block_size for m in coordinator.single_type_managers] == [12288, 1536, 1536, 1536]
    request = SimpleNamespace(
        request_id="seed", block_hashes=[b"s" * 32], num_prompt_tokens=1537, shared_prefix_boundary=0
    )
    coordinator.allocate_new_blocks("seed", 1536, 1536)
    coordinator.cache_blocks(request, 1536)
    coordinator.new_step_starts()
    hits, length, *_ = coordinator.find_longest_cache_hit(request.block_hashes, 1536)
    assert length == 1536
    assert all(len(blocks) == 1 for blocks in hits)
    coordinator.allocate_new_computed_blocks("reuse", hits, 1536, 0)
    coordinator.allocate_new_blocks("reuse", 1537, 1537)
    attention = coordinator.single_type_managers[0]
    assert attention.req_to_blocks["reuse"][0] is not attention.req_to_blocks["seed"][0]
    for state in coordinator.single_type_managers[1:]:
        assert state.req_to_blocks["reuse"][-1] is not state.req_to_blocks["seed"][-1]
