#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import unittest
from unittest.mock import patch
from weakref import ref

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, MambaSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
# isort: on


def _hashes(num_blocks: int) -> list[bytes]:
    return [bytes([idx % 251]) * 32 for idx in range(num_blocks)]


def _full_config(block_size: int):
    spec = AttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32)
    return FullAttentionManager.layer_config(spec)


def _sliding_config(block_size: int, sliding_window: int):
    spec = AttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32)
    return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)


def _group(layer_names, layer_config, **kwargs):
    return KVCacheGroup.from_layers(dict.fromkeys(layer_names, layer_config), **kwargs)


class TestAscendStoreCoordinator(unittest.TestCase):
    def test_compressed_group_hits_on_effective_granularity(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _full_config(128 * 128))],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128 * 128],
            group_cache_families=["c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, {(0, bytes(grouped_hash))}),
        )

        self.assertEqual(hit_length, 128 * 128)

    def test_missing_required_group_returns_zero(self):
        block_hashes = _hashes(128)
        c1_exists = {(0, block_hash) for block_hash in block_hashes}
        coord = AscendStoreCoordinator(
            [
                _group(["layer.0"], _full_config(128)),
                _group(["layer.1"], _full_config(128 * 128)),
            ],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128, 128 * 128],
            group_cache_families=["c1", "c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, c1_exists),
        )

        self.assertEqual(hit_length, 0)
        for local, expected in ((16383, 0), (16384, 16384)):
            with self.subTest(local=local):
                self.assertEqual(
                    coord.find_reachable_hit_tokens(block_hashes, 16384, lambda *args: [], hbm_hit_tokens=local),
                    expected,
                )

    def test_store_mask_uses_manager_reachability(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _sliding_config(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )

        with patch.object(
            SlidingWindowManager,
            "reachable_block_mask",
            return_value=[False, False, False, True],
        ):
            masks = coord.store_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))

    def test_lookup_mask_uses_reachability_without_retention(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _sliding_config(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
            retention_interval=256,
        )
        with patch.object(
            SlidingWindowManager,
            "reachable_block_mask",
            return_value=[False, False, False, True],
        ) as reachable:
            masks = coord.lookup_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))
        self.assertEqual(reachable.call_args.kwargs["retention_interval"], 256)

    def test_mamba_replay_boundary_and_partial_prefix_lookup(self):
        spec = MambaSpec(block_size=256, shapes=((1,),), dtypes=(torch.float32,))
        state_config = MambaManager.layer_config(spec, mamba_cache_mode="align")
        for interval, expected in [
            (0, [False, False, True, False]),
            (512, [False, True, True, True]),
            (None, [True, True, True, True]),
        ]:
            with self.subTest(retention_interval=interval):
                coord = AscendStoreCoordinator(
                    [_group(["attention"], _full_config(256)), _group(["state"], state_config)],
                    scheduler_block_size=256,
                    hash_block_size=256,
                    group_block_sizes=[256, 256],
                    group_cache_families=["default", "default"],
                    retention_interval=interval,
                )
                self.assertEqual(coord.store_mask(1024, num_prompt_tokens=769)[1], expected)
                hashes = _hashes(4)

                def query(group_id, group_hashes, mask):
                    # A shorter request published only its 768-token checkpoint.
                    return [
                        h for i, h in enumerate(group_hashes) if (mask is None or mask[i]) and (group_id == 0 or i == 2)
                    ]

                self.assertEqual(coord.find_reachable_hit_tokens(hashes, 1024, query), 768)

    def test_store_mask_propagates_eagle_to_same_spec_siblings(self):
        calls = []

        def fake_reachable_block_mask(*args, **kwargs):
            calls.append(kwargs["use_eagle"])
            return [True, False, True, False]

        shared_spec = _sliding_config(block_size=128, sliding_window=256)
        coord = AscendStoreCoordinator(
            [
                _group(["layer.0"], shared_spec),
                _group(["layer.mtp"], shared_spec, is_eagle_group=True),
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c1"],
        )

        with patch.object(
            SlidingWindowManager,
            "reachable_block_mask",
            side_effect=fake_reachable_block_mask,
        ):
            masks = coord.store_mask(512)

        self.assertEqual(calls, [True, True])
        self.assertEqual(masks, ([True, False, True, False], [True, False, True, False]))

    def test_reachable_mask_cache_preserves_inputs_and_caller_isolation(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _sliding_config(16, 32))],
            scheduler_block_size=16,
            hash_block_size=16,
            group_block_sizes=[16],
            group_cache_families=["c1"],
        )
        with patch.object(
            SlidingWindowManager,
            "reachable_block_mask",
            side_effect=lambda *args, **kwargs: [True] + [False] * (kwargs["end_block"] - 1),
        ) as build_mask:
            coord.store_mask(32, 33)[0][0] = False
            self.assertEqual(coord.store_mask(32, 33), ([True, False],))
            coord.lookup_mask(32)[0][0] = False
            self.assertEqual(coord.lookup_mask(32), ([True, False],))
            self.assertEqual(build_mask.call_count, 2)
            coord.retention_interval = 64
            coord.store_mask(32, 33)
            coord.store_mask(48, 33)
            coord.store_mask(32, 65)
            self.assertEqual(build_mask.call_count, 5)

        cached_masks = coord._cached_reachable_masks
        coord_ref = ref(coord)
        del coord
        self.assertIsNone(coord_ref())
        self.assertGreater(cached_masks.cache_info().currsize, 0)

    def test_compressed_masks_use_full_attention_reachability(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _full_config(block_size=512))],
            scheduler_block_size=2048,
            hash_block_size=128,
            group_block_sizes=[512],
            group_cache_families=["c4"],
        )

        self.assertEqual(
            coord.store_mask(2048, num_prompt_tokens=2048),
            ([True, True, True, True],),
        )
        cached_mask = [True, True, False, False]
        with patch.object(
            coord,
            "find_longest_cache_hit",
            return_value=((cached_mask,), 1024),
        ):
            self.assertEqual(
                coord.load_mask(_hashes(16), 2048),
                (cached_mask,),
            )


class TestFindReachableHitTokens(unittest.TestCase):
    """The shared driver behind the scheduler/worker coordinator lookups."""

    def test_passes_mask_allowed_hashes_to_query_callback(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _sliding_config(block_size=128, sliding_window=256))],
            scheduler_block_size=256,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )
        calls: list[tuple[int, list, list | None]] = []

        def query_group_hits(group_id, group_block_hashes, lookup_mask):
            calls.append((group_id, list(group_block_hashes), list(lookup_mask) if lookup_mask is not None else None))
            return []

        with patch.object(
            SlidingWindowManager,
            "reachable_block_mask",
            return_value=[False, True],
        ):
            hit = coord.find_reachable_hit_tokens(_hashes(2), 256, query_group_hits)

        self.assertEqual(hit, 0)
        self.assertEqual(calls, [(0, _hashes(2), [False, True])])

    def test_hit_length_derived_from_callback_hits(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _full_config(128))],
            scheduler_block_size=256,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c4"],
        )
        block_hashes = _hashes(4)
        received: list[list] = []

        def query_group_hits(group_id, group_block_hashes, lookup_mask):
            received.append(list(group_block_hashes))
            self.assertIsNone(lookup_mask)
            return list(group_block_hashes[:3])

        with patch.object(coord, "find_longest_cache_hit", wraps=coord.find_longest_cache_hit) as derive:
            hit = coord.find_reachable_hit_tokens(block_hashes, 384, query_group_hits)

        # Only whole hash blocks within token_len are queried (384 // 128 = 3).
        self.assertEqual(received, [_hashes(3)])
        self.assertEqual(hit, 384)
        derive.assert_called_once()
        self.assertFalse(derive.call_args.kwargs["apply_eagle"])

    def test_no_hits_returns_zero_without_deriving(self):
        coord = AscendStoreCoordinator(
            [_group(["layer.0"], _full_config(128))],
            scheduler_block_size=256,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c4"],
        )

        with patch.object(coord, "find_longest_cache_hit") as derive:
            hit = coord.find_reachable_hit_tokens(_hashes(2), 256, lambda *args: [])

        self.assertEqual(hit, 0)
        derive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
