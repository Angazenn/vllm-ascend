"""Unit tests for SFAKVOffloadWorker layer registration.

Covers:
- BF16 offload layer selection by the five-tensor tuple length.

The worker module JIT-builds a C++ extension and imports memfabric_hybrid at
module load time, neither of which is available in the UT sandbox; both are
stubbed before the import below.
"""

from unittest.mock import MagicMock

# Stub heavy module-level dependencies BEFORE importing the worker.
# 1. cpu_sparse_attn cpp extension JIT build (torch.utils.cpp_extension.load).
import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())  # noqa: E402

# 2. memfabric_hybrid.offload is not exported in the sandbox install.
import memfabric_hybrid  # noqa: E402

if not hasattr(memfabric_hybrid, "offload"):  # noqa: E402
    memfabric_hybrid.offload = MagicMock()  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (  # noqa: E402
    SFAKVOffloadWorker,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (  # noqa: E402
    ReqMeta,
    SFAKVOffloadConnectorMetadata,
)


def _make_worker_without_init() -> SFAKVOffloadWorker:
    """Bypass __init__ (heavy); set only the attrs _register_offload_layers reads."""
    w = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    w.num_target_layers = 0
    w.tp_rank = 0
    w.pending_save_layer_ids = set()
    w.submitted_save_layer_ids = set()
    w.completed_cpu_blocks = {}
    w.pending_completed_cpu_blocks = {}
    return w


def _tuple(n: int) -> tuple:
    return tuple(torch.zeros(1) for _ in range(n))


def test_register_selects_offload_tuples_and_skips_others():
    w = _make_worker_without_init()
    kv_caches = {
        "layer.0": _tuple(5),
        "layer.1": _tuple(5),
        "indexer.layer.0": torch.zeros(1),
        "layer.2": _tuple(6),
        "layer.3": _tuple(3),
    }
    w._register_offload_layers(kv_caches)
    assert w.offload_layer_names == ["layer.0", "layer.1"]
    assert w.num_offload_layers == 2


def test_register_raises_when_no_offload_layers():
    w = _make_worker_without_init()
    with pytest.raises(ValueError, match="did not find SFA KV cache layers"):
        w._register_offload_layers({"layer.0": _tuple(3)})


def test_register_all_five_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(5), "layer.1": _tuple(5)})
    assert w.num_offload_layers == 2


def test_cpu_block_count_promotes_only_after_save_completion():
    w = _make_worker_without_init()
    metadata = SFAKVOffloadConnectorMetadata({"req-0"}, set())
    metadata.add_request(
        ReqMeta(
            req_id="req-0",
            block_ids_npu=[10, 11, 12, 13, 14],
            block_ids_cpu=[1, 2, 3, 4, 5],
            num_new_offload_blocks=1,
        )
    )

    w._stage_cpu_block_completion(metadata)

    assert w.get_num_cpu_blocks(["req-0"]) == {"req-0": 4}
    w._commit_cpu_block_completion()
    assert w.get_num_cpu_blocks(["req-0"]) == {"req-0": 5}


def test_cpu_block_count_drops_finished_requests():
    w = _make_worker_without_init()
    w.completed_cpu_blocks["finished"] = 3
    w.pending_completed_cpu_blocks["finished"] = 4

    w._stage_cpu_block_completion(
        SFAKVOffloadConnectorMetadata(set(), set())
    )

    assert w.get_num_cpu_blocks(["finished"]) == {"finished": 0}
    assert w.pending_completed_cpu_blocks == {}
