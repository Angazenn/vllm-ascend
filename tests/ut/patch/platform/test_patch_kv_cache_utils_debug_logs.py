# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import weakref
from collections import OrderedDict
from types import SimpleNamespace

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


def _attach_fake_kv_cache_manager(
    pool: BlockPool,
    req_to_blocks: dict[str, list[KVCacheBlock]] | None = None,
) -> None:
    class FakeKVCacheManager:
        pass

    manager = SimpleNamespace(req_to_blocks=req_to_blocks or {})
    kv_cache_manager = FakeKVCacheManager()
    kv_cache_manager.block_pool = pool
    kv_cache_manager.coordinator = SimpleNamespace(single_type_managers=(manager,))
    setattr(pool, "_test_kv_cache_manager", kv_cache_manager)
    setattr(pool, patch._KV_CACHE_MANAGER_ATTR, weakref.ref(kv_cache_manager))


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


def test_debug_pop_mismatch_warning_is_suppressed_after_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    setattr(queue, patch._DEBUG_QUEUE_ATTR, OrderedDict.fromkeys([2, 1, 3]))

    messages = _collect_warning_messages(monkeypatch)
    queue.popleft()
    queue.popleft()

    warning_text = "\n".join(messages)
    assert warning_text.count("free_queue_debug_pop_mismatch") == 1


def test_debug_count_drift_warning_logs_on_drift_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(1, 4)])
    queue.num_free_blocks = 4

    messages = _collect_warning_messages(monkeypatch)
    for _ in range(10001):
        patch._warn_debug_count_drift(queue, "test")
    queue.num_free_blocks = 8
    patch._warn_debug_count_drift(queue, "test")
    patch._warn_debug_count_drift(queue, "test")
    queue.num_free_blocks = 3
    patch._warn_debug_count_drift(queue, "test")
    queue.num_free_blocks = 2
    patch._warn_debug_count_drift(queue, "test")

    warning_text = "\n".join(messages)
    assert warning_text.count("free_queue_debug_count_mismatch") == 3
    assert "free_block_count_drift=1" in warning_text
    assert "free_block_count_drift=5" in warning_text
    assert "previous_free_block_count_drift=1" in warning_text
    assert "free_block_count_drift=-1" in warning_text
    assert "previous_free_block_count_drift=None" in warning_text


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


def test_get_new_blocks_rebuilds_free_queue_when_ref_cnt_audit_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    _attach_fake_kv_cache_manager(pool)

    first_block = pool.free_block_queue.fake_free_list_head.next_free_block
    assert first_block is not None
    first_block.next_free_block = None
    pool.free_block_queue.num_free_blocks = 4

    messages = _collect_warning_messages(monkeypatch)
    blocks = pool.get_new_blocks(2)

    assert [block.block_id for block in blocks] == [1, 2]
    assert _free_block_ids(pool.free_block_queue) == [3, 4]

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG free_queue_rebuilt" in warning_text
    assert "where=BlockPool.get_new_blocks:exception" in warning_text


def test_get_new_blocks_does_not_rebuild_when_ref_cnt_audit_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    live_block = pool.blocks[1]
    _attach_fake_kv_cache_manager(pool, {"req": [live_block]})

    first_block = pool.free_block_queue.fake_free_list_head.next_free_block
    assert first_block is not None
    first_block.next_free_block = None
    pool.free_block_queue.num_free_blocks = 4

    messages = _collect_warning_messages(monkeypatch)
    with pytest.raises(AssertionError):
        pool.get_new_blocks(2)

    warning_text = "\n".join(messages)
    assert "SWA_BLOCK_DIAG ref_cnt_audit_mismatch" in warning_text
    assert "SWA_BLOCK_DIAG free_queue_rebuild_skipped" in warning_text
    assert "reason=ref_cnt_audit_mismatch" in warning_text
