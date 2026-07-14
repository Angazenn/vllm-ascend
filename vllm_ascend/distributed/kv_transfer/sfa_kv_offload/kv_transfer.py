import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    LayerMultiBlockReqMeta,
    ReqMeta,
)


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        block_size: int | list[int],
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.ready_event = ready_event
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread: %s", e)

    def _handle_request(self, req_meta: Any):
        pass


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        block_size: int | list[int],
        num_layers: int,
        tp_rank: int,
        ready_event: threading.Event,
        layer_save_finished_events: list[threading.Event] = [],
        scratch_save_finished_events: list[threading.Event] | None = None,
    ):
        super().__init__(
            block_size, tp_rank, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.num_layers = num_layers
        self.layer_save_finished_events = layer_save_finished_events
        self.scratch_save_finished_events = scratch_save_finished_events
        self.save_stream = torch_npu.npu.Stream()

    def _copy_requests(
        self,
        req_metas: list[LayerMultiBlockReqMeta],
        layer_id: int,
    ) -> None:
        with torch_npu.npu.stream(self.save_stream):
            for req_meta in req_metas:
                req_id = req_meta.req_id
                block_ids_npu = req_meta.block_ids_npu
                block_ids_cpu = req_meta.block_ids_cpu
                assert block_ids_npu is not None
                assert block_ids_cpu is not None
                assert req_meta.cache_npu is not None
                assert req_meta.cache_cpu is not None
                (k_cache_npu, v_cache_npu) = req_meta.cache_npu
                (k_cache_cpu, v_cache_cpu) = req_meta.cache_cpu
                if len(block_ids_npu) != len(block_ids_cpu):
                    logger.error(
                        "Offload req %s fail! npu block num (%s) "
                        "cpu block num (%s) size mismatch",
                        req_id,
                        len(block_ids_npu),
                        len(block_ids_cpu),
                    )
                if (
                    self.tp_rank == 0
                    and layer_id == 0
                    and envs.VLLM_ASCEND_SFA_DEBUG
                ):
                    logger.info(
                        ">>>>> kv sending thread offload %s blocks of req %s",
                        len(block_ids_npu),
                        req_id,
                    )
                if len(block_ids_npu) > 1:
                    k_cache_cpu[block_ids_cpu] = k_cache_npu[
                        block_ids_npu
                    ].to("cpu")
                    v_cache_cpu[block_ids_cpu] = v_cache_npu[
                        block_ids_npu
                    ].to("cpu")
                else:
                    k_cache_cpu[block_ids_cpu[0]].copy_(
                        k_cache_npu[block_ids_npu[0]]
                    )
                    v_cache_cpu[block_ids_cpu[0]].copy_(
                        v_cache_npu[block_ids_npu[0]]
                    )
        self.save_stream.synchronize()

    def _handle_request(
        self, req_metas: list[LayerMultiBlockReqMeta]
    ):
        # logger.info(f'>>>>> handle request, req_metas = {len(req_metas)}')
        if len(req_metas) == 0:
            return

        layer_id = req_metas[0].layer_id

        scratch_requests = [
            request for request in req_metas if request.uses_shared_scratch
        ]
        tail_requests = [
            request for request in req_metas if not request.uses_shared_scratch
        ]
        if scratch_requests:
            self._copy_requests(scratch_requests, layer_id)
            if self.scratch_save_finished_events is not None:
                self.scratch_save_finished_events[layer_id].set()
        if tail_requests:
            self._copy_requests(tail_requests, layer_id)

        req_metas.clear()
        self.request_queue.task_done()
        self.layer_save_finished_events[layer_id].set()
