import torch
from vllm.v1.core.block_pool import BlockPool

from vllm_ascend.core.kv_cache_interface import make_offload_main_mla_spec
from vllm_ascend.core.single_type_kv_cache_manager import (
    OffloadMLAAttentionManager,
)


BLOCK_SIZE = 128


def _make_manager() -> OffloadMLAAttentionManager:
    spec = make_offload_main_mla_spec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    return OffloadMLAAttentionManager(
        spec,
        block_pool=BlockPool(
            num_gpu_blocks=32,
            enable_caching=False,
            hash_block_size=BLOCK_SIZE,
            enable_kv_cache_events=False,
        ),
        enable_caching=False,
        kv_cache_group_id=1,
        scheduler_block_size=BLOCK_SIZE,
    )


def test_mtp_does_not_free_unfinalized_boundary_block():
    manager = _make_manager()
    req_id = "mtp-request"

    # Model the state immediately before a speculative boundary step. Four
    # prompt blocks are already on CPU; the fifth HBM block is the resident
    # decode tail. A draft made the previous scheduled target length reach 640,
    # but only 639 tokens are known finalized and safe to offload.
    manager.req_to_blocks[req_id] = manager.block_pool.get_new_blocks(1)
    manager.req_to_offloaded_blocks[req_id] = (
        manager.block_pool.get_new_blocks(4)
    )
    tail_block_id = manager.req_to_blocks[req_id][0].block_id
    manager.num_cached_block[req_id] = 5
    manager.req_to_num_allocated_tokens[req_id] = 640
    manager.req_to_num_finalized_tokens[req_id] = 639
    manager.decode_threshold = 2

    manager.get_num_blocks_to_allocate(
        request_id=req_id,
        num_tokens=642,
        new_computed_blocks=[],
        total_computed_tokens=640,
        num_tokens_main_model=641,
    )
    manager.allocate_new_blocks(
        request_id=req_id,
        num_tokens=642,
        num_tokens_main_model=641,
    )

    assert len(manager.req_to_offloaded_blocks[req_id]) == 4
    assert manager.req_to_blocks[req_id][0].block_id == tail_block_id

    # On the following scheduler step, the finalized cursor has crossed 640.
    # The connector had the intervening forward pass to copy the block, so it is
    # now safe for the manager to release it.
    manager.get_num_blocks_to_allocate(
        request_id=req_id,
        num_tokens=643,
        new_computed_blocks=[],
        total_computed_tokens=641,
        num_tokens_main_model=642,
    )
    manager.allocate_new_blocks(
        request_id=req_id,
        num_tokens=643,
        num_tokens_main_model=642,
    )

    assert len(manager.req_to_offloaded_blocks[req_id]) == 5
    assert manager.req_to_offloaded_blocks[req_id][-1].block_id == tail_block_id
