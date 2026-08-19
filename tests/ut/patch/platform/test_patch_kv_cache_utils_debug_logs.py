# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

from vllm_ascend.patch.platform import patch_kv_cache_utils as patch  # noqa: F401


def _free_block_ids(queue: FreeKVCacheBlockQueue) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


def _collect_warning_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []

    def warning(message, *args, **_kwargs) -> None:
        messages.append(str(message) % args if args else str(message))

    monkeypatch.setattr(patch.logger, "warning", warning)
    return messages


def test_append_n_duplicate_warning_includes_debug_queue_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])

    block = queue.popleft()
    queue.append_n([block])

    messages = _collect_warning_messages(monkeypatch)
    queue.append_n([block])

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG debug_duplicate_free_queue_insert" in warning_text
    assert "where=FreeKVCacheBlockQueue.append_n" in warning_text
    assert "block_id=1" in warning_text
    assert "debug_block_ids=[2, 3, 1]" in warning_text
    assert "debug_contains_target=True" in warning_text


def test_popleft_n_exception_warning_includes_debug_expected_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    queue.num_free_blocks = 5

    messages = _collect_warning_messages(monkeypatch)
    with pytest.raises(AssertionError):
        queue.popleft_n(5)

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG free_queue_mismatch" in warning_text
    assert "where=FreeKVCacheBlockQueue.popleft_n:exception" in warning_text
    assert "debug_block_ids=[1, 2, 3]" in warning_text
    assert "debug_expected_pop_block_ids=[1, 2, 3]" in warning_text
    assert "requested_n=5" in warning_text


def test_valid_remove_does_not_warn_missing_remove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    middle_block = queue.fake_free_list_head.next_free_block.next_free_block
    assert middle_block is not None

    messages = _collect_warning_messages(monkeypatch)
    queue.remove(middle_block)

    warning_text = "\n".join(messages)
    assert "free_queue_debug_missing_remove" not in warning_text
    assert _free_block_ids(queue) == [1, 3]


def test_stale_delayed_free_after_reuse_is_not_a_queue_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)

    delayed_req_blocks = pool.get_new_blocks(2)
    delayed_req_block_ids = {block.block_id for block in delayed_req_blocks}

    pool.free_blocks(reversed(delayed_req_blocks))
    pressure_req_blocks = pool.get_new_blocks(4)
    pressure_req_block_ids = {block.block_id for block in pressure_req_blocks}
    assert delayed_req_block_ids <= pressure_req_block_ids

    messages = _collect_warning_messages(monkeypatch)
    pool.free_blocks(reversed(delayed_req_blocks))

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG" not in warning_text
    assert delayed_req_block_ids <= set(_free_block_ids(pool.free_block_queue))


def test_middle_duplicate_insert_fault_shape_reproduces_popleft_n_assert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    middle_block = queue.fake_free_list_head.next_free_block.next_free_block
    assert middle_block is not None

    patch._orig_free_queue_append_n(queue, [middle_block])

    messages = _collect_warning_messages(monkeypatch)
    with pytest.raises(AssertionError):
        queue.popleft_n(queue.num_free_blocks)

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG free_queue_mismatch" in warning_text


def test_middle_duplicate_insert_is_guarded_before_queue_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    middle_block = queue.fake_free_list_head.next_free_block.next_free_block
    assert middle_block is not None

    messages = _collect_warning_messages(monkeypatch)
    queue.append_n([middle_block])

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG debug_duplicate_free_queue_insert" in warning_text
    assert "where=FreeKVCacheBlockQueue.append_n" in warning_text
    assert "block_id=2" in warning_text
    assert _free_block_ids(queue) == [1, 2, 3]
    assert [block.block_id for block in queue.popleft_n(queue.num_free_blocks)] == [
        1,
        2,
        3,
    ]
