from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


_DIAG_LOG_FIRST_N = 5
_DIAG_LOG_EVERY_N = 100
_diag_log_counts: defaultdict[str, int] = defaultdict(int)


def _should_log_diag(kind: str) -> bool:
    _diag_log_counts[kind] += 1
    count = _diag_log_counts[kind]
    return count <= _DIAG_LOG_FIRST_N or count % _DIAG_LOG_EVERY_N == 0


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        offloaded = False
        for c in self._connectors:
            hook = getattr(c, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        connector_results: list[str] = []
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            connector_results.append(
                f"{c.__class__.__name__}:async_save={async_save}:txfer_params={txfer_params is not None}"
            )
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        logger.debug(
            "SWA_BLOCK_DIAG multi_connector_request_finished "
            "req_id=%s async_saves=%d group_block_counts=%s connector_results=%s",
            request.request_id,
            async_saves,
            [len(group_block_ids) for group_block_ids in block_ids],
            connector_results,
        )
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1
            if _should_log_diag("multi_connector_multi_async_save"):
                logger.warning(
                    "SWA_BLOCK_DIAG multi_connector_multi_async_save "
                    "where=AscendMultiConnector.request_finished_all_groups "
                    "req_id=%s async_saves=%d extra_async_saves=%d "
                    "group_block_counts=%s connector_results=%s",
                    request.request_id,
                    async_saves,
                    async_saves - 1,
                    [len(group_block_ids) for group_block_ids in block_ids],
                    connector_results,
                )

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            connector_name = c.__class__.__name__
            logger.debug(
                "SWA_BLOCK_DIAG multi_connector_child_finished "
                "connector=%s sending_count=%d recving_count=%d sending_sample=%s recving_sample=%s",
                connector_name,
                len(sending or ()),
                len(recving or ()),
                list(sending or ())[:8],
                list(recving or ())[:8],
            )
            finished_recving.update(recving or ())
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                    if _should_log_diag("multi_connector_release_after_extra_async_save"):
                        logger.warning(
                            "SWA_BLOCK_DIAG multi_connector_release_after_extra_async_save "
                            "connector=%s req_id=%s",
                            connector_name,
                            req_id,
                        )
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1
                    if _should_log_diag("multi_connector_hold_finished_sending"):
                        logger.warning(
                            "SWA_BLOCK_DIAG multi_connector_hold_finished_sending "
                            "connector=%s req_id=%s extra_pending_before=%d extra_pending_after=%d",
                            connector_name,
                            req_id,
                            extra_pending,
                            extra_pending - 1,
                        )

        return finished_sending or None, finished_recving or None
