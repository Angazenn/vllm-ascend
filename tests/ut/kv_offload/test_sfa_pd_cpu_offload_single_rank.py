"""Regression tests for split-cache SFA PD CPU offload."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest

torch = pytest.importorskip("torch")

import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())

memfabric_hybrid = pytest.importorskip("memfabric_hybrid")
if not hasattr(memfabric_hybrid, "offload"):
    memfabric_hybrid.offload = MagicMock()

from vllm_ascend.core.kv_cache_interface import (  # noqa: E402
    AscendMLAAttentionSpec,
    AscendSFALayerwiseIndexerCacheSpec,
    AscendSFAOffloadIndexerCacheSpec,
    OffloadMLAAttentionSpec,
    get_sfa_layerwise_ascend_store_config,
    uses_split_sfa_decode_offload_layout,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload import (  # noqa: E402
    worker as worker_module,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.layout import (  # noqa: E402
    resolve_sfa_split_cache_layout,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (  # noqa: E402
    MF_META,
    ROLE_INDEXER_K,
    ROLE_MAIN_K,
    ROLE_MAIN_V,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (  # noqa: E402
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (  # noqa: E402
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (  # noqa: E402
    SFAPDCpuOffloadConsumerWorker,
    SFAPDCpuOffloadProducerWorker,
)

MAIN_0 = "model.layers.0.self_attn.attn"
MAIN_1 = "model.layers.1.self_attn.attn"
MAIN_2 = "model.layers.2.self_attn.attn"
INDEXER_0 = "model.layers.0.self_attn.indexer.k_cache"
INDEXER_2 = "model.layers.2.self_attn.indexer.k_cache"


class FakeTensor:
    def __init__(self, shape, ptr: int, element_size: int = 2):
        self.shape = shape
        self._ptr = ptr
        self._element_size = element_size

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return self._element_size


def _group(spec, names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


def _decode_cache_config(group_order=("main", "indexer")):
    main_spec = OffloadMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = AscendSFAOffloadIndexerCacheSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        indexer_pad_dim=16,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    groups = {
        "main": _group(main_spec, [MAIN_0, MAIN_1, MAIN_2]),
        "indexer": _group(indexer_spec, [INDEXER_0, INDEXER_2]),
    }
    return SimpleNamespace(
        kv_cache_groups=[groups[name] for name in group_order],
        num_blocks=16,
        kv_cache_tensors=[],
    )


def _prefill_cache_config():
    main_spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    indexer_spec = AscendSFALayerwiseIndexerCacheSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        indexer_pad_dim=16,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    return SimpleNamespace(
        kv_cache_groups=[
            _group(main_spec, [MAIN_0, MAIN_1, MAIN_2]),
            _group(indexer_spec, [INDEXER_0]),
            _group(indexer_spec, [INDEXER_2]),
        ],
        num_blocks=16,
        kv_cache_tensors=[],
    )


def test_nested_ascend_store_and_pd_decode_layout_detection():
    nested = SimpleNamespace(
        additional_config={"use_offload": False},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "layerwise_num_shared_buffers": 4,
                "connectors": [
                    {
                        "kv_connector": "SFAPDCpuOffloadConnector",
                        "kv_connector_extra_config": {"use_layerwise": True},
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_connector_extra_config": {"use_layerwise": True},
                    },
                ],
            },
        ),
    )
    assert get_sfa_layerwise_ascend_store_config(nested) == {
        "layerwise_num_shared_buffers": 4,
        "use_layerwise": True,
    }

    decode = SimpleNamespace(
        additional_config={"use_offload": True},
        kv_transfer_config=SimpleNamespace(
            kv_connector="SFAPDCpuOffloadConnector",
            kv_connector_extra_config={"use_layerwise": True},
            is_kv_consumer=True,
        ),
    )
    assert uses_split_sfa_decode_offload_layout(decode)


def test_group_resolution_uses_specs_and_keeps_real_indexer_subset():
    config = _decode_cache_config(group_order=("main", "indexer"))
    layout = resolve_sfa_split_cache_layout(config, "decode")

    assert layout.main_group_id == 0
    assert layout.indexer_group_ids == (1,)
    assert layout.main_layer_names == (MAIN_0, MAIN_1, MAIN_2)
    assert layout.indexer_owner_set == frozenset({INDEXER_0, INDEXER_2})
    assert 1 not in layout.indexer_name_by_layer_id


def test_decode_registration_binds_separate_real_indexer_entries():
    config = _decode_cache_config()
    consumer = SFAPDCpuOffloadConsumerWorker.__new__(
        SFAPDCpuOffloadConsumerWorker
    )
    consumer.kv_cache_config = config
    consumer.cache_layout = resolve_sfa_split_cache_layout(config, "decode")
    consumer.layer_metadata = {}
    consumer.sfa_worker = SimpleNamespace(
        layer_name_to_offload_id={MAIN_0: 0, MAIN_1: 1, MAIN_2: 2},
        _get_offload_layer_id=lambda name: {
            MAIN_0: 0,
            MAIN_1: 1,
            MAIN_2: 2,
        }[name],
    )
    consumer._dest_blocks_by_req = {}
    consumer.engine = MagicMock()
    consumer._ensure_engine = MagicMock(return_value=consumer.engine)
    consumer.tp_rank = 0
    consumer.side_channel_port = 1234

    placeholder = FakeTensor((0,), 1)
    main_entries = {
        name: (
            FakeTensor((16, 5), 1000 + layer_id * 100),
            FakeTensor((16, 10), 2000 + layer_id * 100),
            placeholder,
            FakeTensor((1, 1), 3000 + layer_id * 100),
            FakeTensor((1, 1), 4000 + layer_id * 100),
        )
        for layer_id, name in enumerate((MAIN_0, MAIN_1, MAIN_2))
    }
    real_indexer_0 = FakeTensor((64, 2, 1, 128), 8000)
    real_indexer_2 = FakeTensor((64, 2, 1, 128), 9000)
    kv_caches = {
        **main_entries,
        INDEXER_0: (real_indexer_0,),
        INDEXER_2: (real_indexer_2,),
    }
    consumer._hbm_kv = {
        name: (entry[0], entry[1]) for name, entry in main_entries.items()
    }
    cpu_k = [FakeTensor((32, 5), 10000 + i * 100) for i in range(3)]
    cpu_v = [FakeTensor((32, 10), 11000 + i * 100) for i in range(3)]
    read_thread = MagicMock()
    read_thread.ready_event = MagicMock()

    with patch.object(
        worker_module, "MembPullReadThread", return_value=read_thread
    ):
        consumer._register_memfabric_pull(kv_caches, cpu_k, cpu_v)

    assert consumer._indexer_by_main[MAIN_0][1] is real_indexer_0
    assert MAIN_1 not in consumer._indexer_by_main
    assert consumer._indexer_by_main[MAIN_2][1] is real_indexer_2
    assert all(
        entry[1] is not placeholder
        for entry in consumer._indexer_by_main.values()
    )


def _make_thread(partial_hbm_bid: int | None = 9):
    hbm_k = FakeTensor((16, 5), 5000)
    hbm_v = FakeTensor((16, 10), 6000)
    indexer = FakeTensor((64, 2, 1, 128), 8000)
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={MAIN_0: 0},
        cpu_pools=[
            (FakeTensor((32, 5), 3000), FakeTensor((32, 10), 4000))
        ],
        hbm_kv={MAIN_0: (hbm_k, hbm_v)},
        indexer_by_main={MAIN_0: (INDEXER_0, indexer, None)},
        expected_indexer_owner_names=frozenset({INDEXER_0}),
        dest_blocks_by_req={
            "req-0": ([7], [3, 4], 2, partial_hbm_bid)
        },
        get_offload_layer_id=lambda _: 0,
        num_manager_blocks=16,
    )
    return thread


def _layer(with_indexer=True, cpu_pool=True):
    sources = {
        ROLE_MAIN_K: {"base": 1000, "page_len": 10, "group_id": 2},
        ROLE_MAIN_V: {"base": 2000, "page_len": 20, "group_id": 2},
    }
    if with_indexer:
        sources[ROLE_INDEXER_K] = {
            "base": 7000,
            "page_len": 2048,
            "group_id": 0,
        }
    return {
        "layer_name": MAIN_0,
        "pool_idx": 0,
        "offload_id": 0,
        "sources": sources,
        "cpu_pool": (
            (FakeTensor((32, 5), 3000), FakeTensor((32, 10), 4000))
            if cpu_pool
            else None
        ),
        "hbm_kv": (FakeTensor((16, 5), 5000), FakeTensor((16, 10), 6000)),
        "indexer_entry": (
            (INDEXER_0, FakeTensor((64, 2, 1, 128), 8000), None)
            if with_indexer
            else None
        ),
    }


def test_manager_page_addresses_use_distinct_main_and_indexer_groups():
    thread = _make_thread()
    local, peer, _, info = thread._build_req_descriptors(
        _layer(),
        "req-0",
        [[11], [], [1, 2, 3]],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 3
    assert info["n_indexer"] == 1
    assert 3030 in local and 4060 in local
    assert 5090 in local and 6180 in local
    assert 8000 + 7 * 2048 in local
    assert 7000 + 11 * 2048 in peer


def test_non_tp0_reads_partial_and_indexer_but_not_full_main():
    thread = _make_thread()
    local, _, _, info = thread._build_req_descriptors(
        _layer(cpu_pool=False),
        "req-0",
        [[11], [], [1, 2, 3]],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 1
    assert 3030 not in local and 4060 not in local
    assert 5090 in local and 6180 in local
    assert 8000 + 7 * 2048 in local


def test_skip_topk_layer_transfers_main_only():
    thread = _make_thread(partial_hbm_bid=9)
    thread._state.indexer_by_main = {}
    thread._state.dest_blocks_by_req["req-0"] = ([], [3, 4], 2, 9)
    local, _, _, info = thread._build_req_descriptors(
        _layer(with_indexer=False),
        "req-0",
        [[], [], [1, 2, 3]],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 3
    assert info["n_indexer"] == 0
    assert all(address < 8000 for address in local)


@pytest.mark.parametrize(
    ("p_owners", "missing", "extra"),
    [
        ([], [INDEXER_0], []),
        ([INDEXER_0, INDEXER_2], [], [INDEXER_2]),
    ],
)
def test_mf_meta_requires_exact_indexer_owner_set(
    p_owners, missing, extra
):
    thread = _make_thread()
    payload = msgspec.msgpack.encode(
        {
            MAIN_0: {
                "base_addrs": [1, 2, 3],
                "block_len": [10, 20, 512],
                "block_size_scale": [1, 1, 4],
                "group_ids": [2, 2, 0],
                "roles": [ROLE_MAIN_K, ROLE_MAIN_V, ROLE_INDEXER_K],
            }
        }
    )
    with pytest.raises(RuntimeError) as error:
        thread._accept_p_metadata((MF_META, "p-session", payload, p_owners))
    assert f"missing_on_p={missing}" in str(error.value)
    assert f"extra_on_p={extra}" in str(error.value)


def test_producer_reuse_mates_follow_physical_shared_by_pools():
    worker = SFAPDCpuOffloadProducerWorker.__new__(
        SFAPDCpuOffloadProducerWorker
    )
    worker.cache_layout = resolve_sfa_split_cache_layout(
        _prefill_cache_config(), "prefill"
    )
    physical = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(shared_by=[MAIN_0, MAIN_2, INDEXER_0]),
            SimpleNamespace(shared_by=[MAIN_1, INDEXER_2]),
        ]
    )

    worker.register_kv_cache_config(physical)

    assert worker.get_reuse_mate(2) == 0
    assert worker.get_reuse_mate(1) is None


def test_producer_scheduler_keeps_all_block_groups_and_finishes_chunk():
    scheduler = SFAPDProducerScheduler.__new__(SFAPDProducerScheduler)
    scheduler.block_size = [512, 512, 128]
    scheduler._reqs_need_send_layerwise = {}
    request = SimpleNamespace(
        request_id="req-0-internal",
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_cached_tokens": 0,
            "remote_host": "127.0.0.1",
            "remote_port": 1234,
        },
        all_token_ids=list(range(128)),
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7], [8], [3, 4]))

    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        scheduled_new_reqs=[
            SimpleNamespace(req_id=request.request_id, num_computed_tokens=0)
        ],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={request.request_id: 128},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    req_meta = metadata.requests[request.request_id]
    assert req_meta.local_block_ids == [[7], [8], [3, 4]]
    assert req_meta.chunk_finish is True


def test_producer_worker_preserves_transfer_timeout_setup(monkeypatch):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"transfer_backend": "memfabric"},
            kv_port=14579,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0, tensor_parallel_size=1
        ),
        model_config=SimpleNamespace(use_mla=True),
    )
    engine = MagicMock()
    layout = SimpleNamespace(main_layer_names=[MAIN_0])
    monkeypatch.delenv("ASCEND_TRANSFER_TIMEOUT", raising=False)

    with (
        patch.object(
            worker_module, "resolve_sfa_split_cache_layout", return_value=layout
        ),
        patch.object(worker_module, "get_transfer_timeout_value", return_value=4321),
        patch.object(
            worker_module, "get_tensor_model_parallel_rank", return_value=0
        ),
        patch.object(
            worker_module.torch,
            "npu",
            SimpleNamespace(current_device=MagicMock(return_value=0)),
            create=True,
        ),
        patch.object(worker_module.global_te, "configure"),
        patch.object(
            worker_module.global_te,
            "get_transfer_engine",
            return_value=engine,
        ),
        patch.object(worker_module, "set_shared_layer_transfer_events"),
        patch.object(worker_module, "set_shared_layer_transfer_pending_events"),
    ):
        SFAPDCpuOffloadProducerWorker(
            config, SimpleNamespace(kv_cache_groups=[]), "engine-0"
        )

    assert worker_module.os.environ["ASCEND_TRANSFER_TIMEOUT"] == "4321"
