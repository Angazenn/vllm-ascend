from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class RequestTracker:
    req_id: str
    allocated_block_ids_npu: list[int]
    allocated_block_ids_cpu: list[int]
    prompt_handoff_done: bool = False

    def update(
        self,
        new_block_ids_npu: list[int],
        new_block_ids_cpu: list[int],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        self.allocated_block_ids_npu.extend(new_block_ids_npu)
        self.allocated_block_ids_cpu.extend(new_block_ids_cpu)


@dataclass
class ReqMeta:
    req_id: str
    block_ids_npu: list[int]
    block_ids_cpu: list[int]
    num_new_offload_blocks: int = 0
    num_prompt_blocks: int = 0
    num_transition_prompt_blocks: int = 0
    copy_prompt_tail_to_decode: bool = False

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        num_new_offload_blocks: int = 0,
        num_prompt_blocks: int = 0,
        num_transition_prompt_blocks: int = 0,
        copy_prompt_tail_to_decode: bool = False,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        return ReqMeta(
            req_id=tracker.req_id,
            block_ids_npu=tracker.allocated_block_ids_npu,
            block_ids_cpu=tracker.allocated_block_ids_cpu,
            num_new_offload_blocks=num_new_offload_blocks,
            num_prompt_blocks=num_prompt_blocks,
            num_transition_prompt_blocks=num_transition_prompt_blocks,
            copy_prompt_tail_to_decode=copy_prompt_tail_to_decode,
        )


class SFAKVOffloadConnectorMetadata(KVConnectorMetadata):
    def __init__(
            self,
            unfinished_request_ids: set[str],
            preempted_req_ids: set[str] | None,
        ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    layer_id: int
    block_ids_npu: list[int] | None = None
    block_ids_cpu: list[int] | None = None
    cache_npu: tuple[torch.Tensor, torch.Tensor] | None = None
    cache_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
    use_callback_cache: bool = False
    ready_event: Any | None = None


@dataclass
class LayerPromptTailReqMeta:
    req_id: str
    layer_id: int
    block_id_npu: int
    block_id_decode: int
