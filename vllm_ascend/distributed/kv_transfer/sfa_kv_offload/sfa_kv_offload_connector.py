from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import SFAKVOffloadlScheduler
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import SFAKVOffloadWorker
from vllm_ascend.worker.prefill_decode_kv_cache import get_decode_kv_cache


def _normalize_group_id(group_id: int, num_groups: int) -> int:
    if group_id < 0:
        group_id += num_groups
    if group_id < 0 or group_id >= num_groups:
        raise ValueError(
            f"decode_offload_group_id is out of range: {group_id} for {num_groups} KV cache groups"
        )
    return group_id


def _select_decode_offload_group_id(
    kv_cache_config: KVCacheConfig | None,
    extra_config: dict[str, Any],
) -> int | None:
    if kv_cache_config is None:
        return None
    num_groups = len(kv_cache_config.kv_cache_groups)
    if num_groups <= 1:
        return None
    return _normalize_group_id(
        int(extra_config.get("decode_offload_group_id", -1)),
        num_groups,
    )


def _build_local_kv_cache_config(
    kv_cache_config: KVCacheConfig | None,
    selected_group_id: int | None,
) -> KVCacheConfig | None:
    if kv_cache_config is None or selected_group_id is None:
        return kv_cache_config

    selected_group = kv_cache_config.kv_cache_groups[selected_group_id]
    selected_layers = set(selected_group.layer_names)
    local_tensors: list[KVCacheTensor] = []
    for tensor in kv_cache_config.kv_cache_tensors:
        shared_by = [name for name in tensor.shared_by if name in selected_layers]
        if shared_by:
            local_tensors.append(KVCacheTensor(size=tensor.size, shared_by=shared_by))

    return KVCacheConfig(
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_tensors=local_tensors,
        kv_cache_groups=[selected_group],
    )


class SFAKVOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.use_layerwise = extra_config.get("use_layerwise", False)
        self._selected_kv_cache_group_id = _select_decode_offload_group_id(
            kv_cache_config,
            extra_config,
        )
        self._local_kv_cache_config = _build_local_kv_cache_config(
            kv_cache_config,
            self._selected_kv_cache_group_id,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SFAKVOffloadlScheduler(
                vllm_config,
                self.use_layerwise,
                self._local_kv_cache_config,
                self._selected_kv_cache_group_id,
            )
        else:
            self.connector_worker = SFAKVOffloadWorker(
                vllm_config,
                self.use_layerwise,
                self._local_kv_cache_config,
            )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        # TODO support prefix cache
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request,
            self._select_blocks(blocks),
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        if self._selected_kv_cache_group_id is None:
            return self.request_finished(request, block_ids[-1])
        selected_group_id = self._selected_kv_cache_group_id
        selected_block_ids = (
            block_ids[selected_group_id]
            if selected_group_id < len(block_ids)
            else []
        )
        return self.request_finished(request, selected_block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def _select_blocks(self, blocks: "KVCacheBlocks") -> "KVCacheBlocks":
        if self._selected_kv_cache_group_id is None:
            return blocks
        selected_group_id = self._selected_kv_cache_group_id
        if selected_group_id >= len(blocks.blocks):
            return KVCacheBlocks(([],))
        return KVCacheBlocks((blocks.blocks[selected_group_id],))

    def _filter_decode_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        local_config = self._local_kv_cache_config
        if local_config is None:
            return {
                layer_name: get_decode_kv_cache(cache)
                for layer_name, cache in kv_caches.items()
            }
        layer_names = set(local_config.kv_cache_groups[0].layer_names)
        return {
            layer_name: get_decode_kv_cache(cache)
            for layer_name, cache in kv_caches.items()
            if layer_name in layer_names
        }

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(
            self._filter_decode_kv_caches(kv_caches)
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # In sfa kv offload, we use prepare_lru_resident_and_load instead of wait_for_layer_load
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        self.connector_worker.save_kv_layer()

    def wait_for_save(self):
        self.connector_worker.wait_for_save()

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        capturing: bool = False,
    ) -> bool:
        return self.connector_worker.prepare_lru_resident_and_load(
            layer_name,
            num_reqs,
            topk_indices,
            current_slots,
            req_ids,
            capturing,
        )

    def set_req_ids(self, req_ids: list):
        return self.connector_worker.set_req_ids(req_ids)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # In sfa kv offload, we don't need delay free, thus no need to return finished_send/recv too.
        return (set(), set())
