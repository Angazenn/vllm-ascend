#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


class _FakeBlockTable:
    def __init__(self, block_table: torch.Tensor):
        self._block_table = block_table

    def get_device_tensor(self) -> torch.Tensor:
        return self._block_table


def test_owner_offload_mtp_populates_both_group_mappings():
    proposer = AscendSpecDecodeBaseProposer.__new__(
        AscendSpecDecodeBaseProposer
    )
    proposer.slot_mapping_group = [
        torch.empty(5, dtype=torch.int32)
    ]
    proposer.indexer_slot_mapping_group = [
        torch.empty(5, dtype=torch.int32)
    ]
    proposer.positions = torch.tensor([0, 128, 129])

    groups = [
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=512)
        ),
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=128)
        ),
    ]
    proposer.runner = SimpleNamespace(
        sfa_offload_shared_cache_plan=SimpleNamespace(
            indexer_group_id=0,
            main_group_id=1,
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        input_batch=SimpleNamespace(
            block_table=[
                _FakeBlockTable(torch.tensor([[10, 11]])),
                _FakeBlockTable(torch.tensor([[20, 21, 22]])),
            ]
        ),
        token_to_req=SimpleNamespace(
            gpu=torch.tensor([0, 0, 0], dtype=torch.int32)
        ),
        tail_req_indices=SimpleNamespace(
            gpu=torch.tensor([3], dtype=torch.int32)
        ),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        num_reqs=1,
        positions=torch.tensor([0, 128, 129]),
        token_to_req=torch.tensor([0, 0, 0], dtype=torch.int32),
        tail_req_indices=None,
        indexer_block_table_tensor=torch.empty(0),
        indexer_slot_mapping=torch.empty(0),
    )

    proposer._populate_owner_offload_group_metadata(
        metadata,
        num_input_tokens=5,
    )

    assert torch.equal(
        metadata.slot_mappings_by_group[0],
        torch.tensor([5120, 5248, 5249, -1, -1]),
    )
    assert torch.equal(
        metadata.slot_mappings_by_group[1],
        torch.tensor([2560, 2688, 2689, -1, -1]),
    )
    assert metadata.block_table_tensor.data_ptr() == (
        metadata.block_table_tensors_by_group[1].data_ptr()
    )
    assert metadata.indexer_block_table_tensor is None
    assert metadata.indexer_slot_mapping is None
    assert torch.equal(metadata.tail_req_indices, torch.tensor([3]))


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()
