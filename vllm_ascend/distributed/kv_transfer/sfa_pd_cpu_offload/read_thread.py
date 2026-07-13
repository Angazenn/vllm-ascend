# SPDX-License-Identifier: Apache-2.0
"""D-side memfabric pull read thread for split-cache SFA PD offload."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    GET_META_MSG,
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    ROLE_INDEXER_K,
    ROLE_INDEXER_SCALE,
    ROLE_MAIN_K,
    ROLE_MAIN_V,
    SfaPDAgentMetadata,
)


@dataclass
class ConsumerReadState:
    layer_metadata: dict[str, Any]
    main_name_to_idx: dict[str, int]
    cpu_pools: list[tuple[Any, Any] | None]
    hbm_kv: dict[str, tuple[Any, Any]]
    indexer_by_main: dict[str, tuple[str, Any, Any | None]]
    expected_indexer_owner_names: frozenset[str]
    dest_blocks_by_req: dict[str, tuple[list[int], list[int], int, int | None]]
    get_offload_layer_id: Callable[[str], int]
    num_manager_blocks: int


def _coalesce_desc(
    peer: np.ndarray,
    local: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = peer.shape[0]
    if n <= 1:
        return peer, local, length
    contiguous = (peer[1:] == peer[:-1] + length[:-1]) & (
        local[1:] == local[:-1] + length[:-1]
    )
    if contiguous.all():
        return peer[:1], local[:1], np.array([int(length.sum())], dtype=np.int64)
    run_start = np.concatenate(([0], np.nonzero(~contiguous)[0] + 1))
    run_end = np.append(run_start[1:] - 1, n - 1)
    cumulative = np.cumsum(length)
    merged_len = cumulative[run_end] - cumulative[run_start] + length[run_start]
    return peer[run_start], local[run_start], merged_len


def _tensor_manager_page(tensor: Any, num_manager_blocks: int) -> tuple[int, int]:
    if num_manager_blocks <= 0 or tensor.shape[0] % num_manager_blocks != 0:
        raise ValueError(
            "KV tensor leading dimension must be an integer multiple of "
            f"num_blocks: shape={tuple(tensor.shape)}, num_blocks={num_manager_blocks}"
        )
    block_len = tensor.element_size() * math.prod(tensor.shape[1:])
    block_size_scale = tensor.shape[0] // num_manager_blocks
    return tensor.data_ptr(), block_len * block_size_scale


class MembPullReadThread(threading.Thread):
    """Pull P manager pages into D CPU/HBM destinations by semantic role."""

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        engine: Any,
        state: ConsumerReadState,
    ):
        super().__init__(daemon=True, name=f"MembPullReadThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self.engine = engine
        self._state = state
        self.ready_event = threading.Event()
        self._p_session: str | None = None
        self._p_layer_meta: dict[str, Any] = {}
        self._done_requests: set[str] = set()
        self._lock = threading.Lock()
        self._host = get_ip()

    def get_and_clear_done(self) -> set[str]:
        with self._lock:
            done = self._done_requests
            self._done_requests = set()
            return done

    def _accept_p_metadata(self, msg: tuple) -> None:
        if len(msg) < 4:
            raise RuntimeError("Split-cache SFAPD MF_META is missing indexer owners")
        p_owner_names = frozenset(msg[3])
        expected = self._state.expected_indexer_owner_names
        if p_owner_names != expected:
            missing_on_p = sorted(expected - p_owner_names)
            extra_on_p = sorted(p_owner_names - expected)
            raise RuntimeError(
                "SFA PD real-indexer owner mismatch: "
                f"missing_on_p={missing_on_p}, extra_on_p={extra_on_p}"
            )
        p_layer_meta = msgspec.msgpack.decode(msg[2])
        for layer_name, layer_meta in p_layer_meta.items():
            field_lengths = {
                len(layer_meta.get("base_addrs", [])),
                len(layer_meta.get("block_len", [])),
                len(layer_meta.get("block_size_scale", [])),
                len(layer_meta.get("group_ids", [])),
                len(layer_meta.get("roles", [])),
            }
            if len(field_lengths) != 1:
                raise RuntimeError(
                    f"Malformed P layer metadata for {layer_name}: "
                    f"field lengths={sorted(field_lengths)}"
                )
            roles = layer_meta["roles"]
            if len(roles) != len(set(roles)):
                raise RuntimeError(
                    f"Duplicate tensor roles in P layer metadata for {layer_name}: {roles}"
                )
        self._p_session = msg[1]
        self._p_layer_meta = p_layer_meta

    def run(self):
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self._host, handshake_port)
        logger.info("MembPull read thread listening on: %s", path)
        ctx = zmq.Context()
        try:
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.ROUTER, bind=True
            )
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            encoder = msgspec.msgpack.Encoder()
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    payload = [frame for frame in frames[1:] if frame != b""]
                    if len(payload) != 1:
                        continue
                    msg = decoder.decode(payload[0])
                    msg_type = msg[0]

                    if msg_type == MF_META:
                        try:
                            self._accept_p_metadata(msg)
                        except Exception as error:
                            sock.send_multipart(
                                (
                                    identity,
                                    b"",
                                    encoder.encode((READ_FAILED, -1, str(error))),
                                )
                            )
                            raise
                        logger.info(
                            "Received split MF_META: P session=%s, %d main layers, "
                            "%d real indexer owners",
                            self._p_session,
                            len(self._p_layer_meta),
                            len(self._state.expected_indexer_owner_names),
                        )
                        if envs.VLLM_ASCEND_SFA_DEBUG:
                            for layer_name, layer_meta in self._p_layer_meta.items():
                                logger.info(
                                    "MembPull D recv MF_META layer=%s: roles=%s, "
                                    "groups=%s, block_len=%s, scales=%s",
                                    layer_name,
                                    layer_meta["roles"],
                                    layer_meta["group_ids"],
                                    layer_meta["block_len"],
                                    layer_meta["block_size_scale"],
                                )
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == READ_READY_BATCH:
                        layer_idx = msg[1]
                        layer_name = msg[2]
                        read_reqs = [
                            (
                                entry[0],
                                [list(group) for group in entry[1]],
                            )
                            for entry in msg[3]
                        ]
                        done_ext_ids = list(msg[4]) if len(msg) > 4 else []
                        try:
                            if read_reqs:
                                self._do_read_batch(layer_name, read_reqs)
                            sock.send_multipart(
                                (
                                    identity,
                                    b"",
                                    encoder.encode((READ_DONE, layer_idx)),
                                )
                            )
                        except Exception as error:
                            logger.error(
                                "MembPull batch read failed for layer %d (%s), "
                                "reqs=%d: %s",
                                layer_idx,
                                layer_name,
                                len(read_reqs),
                                error,
                            )
                            sock.send_multipart(
                                (
                                    identity,
                                    b"",
                                    encoder.encode(
                                        (READ_FAILED, layer_idx, str(error))
                                    ),
                                )
                            )
                        if done_ext_ids:
                            with self._lock:
                                self._done_requests.update(done_ext_ids)

                    elif msg_type == GET_META_MSG:
                        meta_bytes = encoder.encode(
                            SfaPDAgentMetadata(
                                te_rpc_port=0,
                                layer_metadata=self._state.layer_metadata,
                            )
                        )
                        sock.send_multipart((identity, b"", meta_bytes))
                    else:
                        logger.error("MembPull got unexpected message %s", msg)
                except Exception as error:
                    logger.error(
                        "MembPull exception: %s: %s", type(error).__name__, error
                    )
        finally:
            ctx.destroy(linger=0)

    def _resolve_read_layer(self, layer_name: str) -> dict[str, Any]:
        state = self._state
        pool_idx = state.main_name_to_idx.get(layer_name)
        if pool_idx is None:
            raise RuntimeError(f"D has no main cache entry for {layer_name}")
        p_meta = self._p_layer_meta.get(layer_name)
        if p_meta is None:
            raise RuntimeError(f"P MF_META has no main layer {layer_name}")

        sources = {}
        for role, base, block_len, scale, group_id in zip(
            p_meta["roles"],
            p_meta["base_addrs"],
            p_meta["block_len"],
            p_meta["block_size_scale"],
            p_meta["group_ids"],
        ):
            sources[role] = {
                "base": base,
                "page_len": block_len * scale,
                "group_id": group_id,
                "block_len": block_len,
                "block_size_scale": scale,
            }
        if ROLE_MAIN_K not in sources or ROLE_MAIN_V not in sources:
            raise RuntimeError(
                f"P layer {layer_name} does not expose main K/V roles"
            )

        offload_id = state.get_offload_layer_id(layer_name)
        cpu_pool = state.cpu_pools[offload_id]
        hbm_k, hbm_v = state.hbm_kv[layer_name]
        indexer_entry = state.indexer_by_main.get(layer_name)
        p_has_indexer = ROLE_INDEXER_K in sources
        if p_has_indexer != (indexer_entry is not None):
            raise RuntimeError(
                f"P/D indexer role mismatch for main layer {layer_name}: "
                f"p_has_indexer={p_has_indexer}, d_has_indexer={indexer_entry is not None}"
            )
        if indexer_entry is not None:
            _, _, d_scale = indexer_entry
            if (ROLE_INDEXER_SCALE in sources) != (d_scale is not None):
                raise RuntimeError(
                    f"P/D indexer scale role mismatch for {layer_name}"
                )

        return {
            "layer_name": layer_name,
            "pool_idx": pool_idx,
            "offload_id": offload_id,
            "sources": sources,
            "cpu_pool": cpu_pool,
            "hbm_kv": (hbm_k, hbm_v),
            "indexer_entry": indexer_entry,
        }

    @staticmethod
    def _source_block_ids(
        block_ids_by_group: list[list[int]], source: dict[str, int]
    ) -> list[int]:
        group_id = source["group_id"]
        if group_id >= len(block_ids_by_group):
            raise RuntimeError(
                f"P READ_READY has no block table for source group {group_id}"
            )
        return block_ids_by_group[group_id]

    @staticmethod
    def _append_pages(
        peer_chunks: list[np.ndarray],
        local_chunks: list[np.ndarray],
        length_chunks: list[np.ndarray],
        source: dict[str, int],
        source_block_ids: list[int],
        local_base: int,
        local_page_len: int,
        local_block_ids: list[int],
    ) -> None:
        if source["page_len"] != local_page_len:
            raise RuntimeError(
                "P/D manager-page size mismatch: "
                f"source={source['page_len']}, destination={local_page_len}"
            )
        if len(source_block_ids) < len(local_block_ids):
            raise RuntimeError(
                "P source block table is shorter than D destination table: "
                f"source={len(source_block_ids)}, destination={len(local_block_ids)}"
            )
        if not local_block_ids:
            return
        source_ids = np.asarray(
            source_block_ids[: len(local_block_ids)], dtype=np.int64
        )
        destination_ids = np.asarray(local_block_ids, dtype=np.int64)
        lengths = np.full(
            len(local_block_ids), source["page_len"], dtype=np.int64
        )
        peer, local, lengths = _coalesce_desc(
            source["base"] + source_ids * source["page_len"],
            local_base + destination_ids * local_page_len,
            lengths,
        )
        peer_chunks.append(peer)
        local_chunks.append(local)
        length_chunks.append(lengths)

    def _build_req_descriptors(
        self,
        layer: dict[str, Any],
        ext_req_id: str,
        p_block_ids_by_group: list[list[int]],
        want_info: bool,
    ) -> tuple[list[int], list[int], list[int], dict[str, Any] | None]:
        state = self._state
        destination = state.dest_blocks_by_req.get(ext_req_id)
        if destination is None:
            raise RuntimeError(f"D has no destination blocks for request {ext_req_id}")
        d_indexer_ids, d_main_cpu_ids, num_full, partial_hbm_bid = destination
        sources = layer["sources"]

        peer_chunks: list[np.ndarray] = []
        local_chunks: list[np.ndarray] = []
        length_chunks: list[np.ndarray] = []

        main_k_source = sources[ROLE_MAIN_K]
        main_v_source = sources[ROLE_MAIN_V]
        p_main_ids = self._source_block_ids(
            p_block_ids_by_group, main_k_source
        )
        if main_v_source["group_id"] != main_k_source["group_id"]:
            raise RuntimeError("P main K/V roles must use the same cache group")

        cpu_pool = layer["cpu_pool"]
        n_main = 0
        if cpu_pool is not None:
            if len(d_main_cpu_ids) != num_full:
                raise RuntimeError(
                    "D CPU destination count does not match full main pages: "
                    f"destinations={len(d_main_cpu_ids)}, num_full={num_full}"
                )
            if d_main_cpu_ids:
                cpu_k, cpu_v = cpu_pool
                cpu_k_page = cpu_k.element_size() * math.prod(cpu_k.shape[1:])
                cpu_v_page = cpu_v.element_size() * math.prod(cpu_v.shape[1:])
                full_ids = p_main_ids[:num_full]
                destination_ids = d_main_cpu_ids[:num_full]
                self._append_pages(
                    peer_chunks,
                    local_chunks,
                    length_chunks,
                    main_k_source,
                    full_ids,
                    cpu_k.data_ptr(),
                    cpu_k_page,
                    destination_ids,
                )
                self._append_pages(
                    peer_chunks,
                    local_chunks,
                    length_chunks,
                    main_v_source,
                    full_ids,
                    cpu_v.data_ptr(),
                    cpu_v_page,
                    destination_ids,
                )
                n_main += len(destination_ids)
        elif num_full:
            # Non-TP0 ranks intentionally have no shared CPU destination.
            d_main_cpu_ids = []

        if partial_hbm_bid is not None:
            if len(p_main_ids) <= num_full:
                raise RuntimeError(
                    f"P has no partial main page for request {ext_req_id}"
                )
            hbm_k, hbm_v = layer["hbm_kv"]
            hbm_k_base, hbm_k_page = _tensor_manager_page(
                hbm_k, state.num_manager_blocks
            )
            hbm_v_base, hbm_v_page = _tensor_manager_page(
                hbm_v, state.num_manager_blocks
            )
            partial_source = [p_main_ids[num_full]]
            self._append_pages(
                peer_chunks,
                local_chunks,
                length_chunks,
                main_k_source,
                partial_source,
                hbm_k_base,
                hbm_k_page,
                [partial_hbm_bid],
            )
            self._append_pages(
                peer_chunks,
                local_chunks,
                length_chunks,
                main_v_source,
                partial_source,
                hbm_v_base,
                hbm_v_page,
                [partial_hbm_bid],
            )
            n_main += 1

        n_indexer = 0
        indexer_tensor = None
        indexer_entry = layer["indexer_entry"]
        if indexer_entry is not None:
            _, indexer_tensor, indexer_scale_tensor = indexer_entry
            indexer_source = sources[ROLE_INDEXER_K]
            p_indexer_ids = self._source_block_ids(
                p_block_ids_by_group, indexer_source
            )
            d_indexer_base, d_indexer_page = _tensor_manager_page(
                indexer_tensor, state.num_manager_blocks
            )
            self._append_pages(
                peer_chunks,
                local_chunks,
                length_chunks,
                indexer_source,
                p_indexer_ids,
                d_indexer_base,
                d_indexer_page,
                d_indexer_ids,
            )
            n_indexer = len(d_indexer_ids)
            if indexer_scale_tensor is not None:
                scale_source = sources[ROLE_INDEXER_SCALE]
                p_scale_ids = self._source_block_ids(
                    p_block_ids_by_group, scale_source
                )
                d_scale_base, d_scale_page = _tensor_manager_page(
                    indexer_scale_tensor, state.num_manager_blocks
                )
                self._append_pages(
                    peer_chunks,
                    local_chunks,
                    length_chunks,
                    scale_source,
                    p_scale_ids,
                    d_scale_base,
                    d_scale_page,
                    d_indexer_ids,
                )

        if not peer_chunks:
            return [], [], [], None
        peer_ptrs = np.concatenate(peer_chunks).tolist()
        local_ptrs = np.concatenate(local_chunks).tolist()
        lengths = np.concatenate(length_chunks).tolist()
        info = None
        if want_info:
            info = {
                "layer_name": layer["layer_name"],
                "ext_req_id": ext_req_id,
                "pool_idx": layer["pool_idx"],
                "offload_id": layer["offload_id"],
                "d_main_ids": d_main_cpu_ids,
                "d_indexer_ids": d_indexer_ids,
                "partial_hbm_bid": partial_hbm_bid,
                "n_main": n_main,
                "n_indexer": n_indexer,
                "num_transfers": len(local_ptrs),
                "indexer_tensor": indexer_tensor,
            }
        return local_ptrs, peer_ptrs, lengths, info

    def _log_read_result(self, read_info: dict[str, Any]) -> None:
        state = self._state
        layer_name = read_info["layer_name"]
        if envs.VLLM_ASCEND_MF_VERIFY:
            try:
                cpu_pool = state.cpu_pools[read_info["offload_id"]]
                if cpu_pool is None:
                    main_k_sum = main_v_sum = 0.0
                else:
                    cpu_k, cpu_v = cpu_pool
                    main_ids = read_info["d_main_ids"]
                    main_k_sum = (
                        cpu_k[main_ids].float().sum().item() if main_ids else 0.0
                    )
                    main_v_sum = (
                        cpu_v[main_ids].float().sum().item() if main_ids else 0.0
                    )
                partial_bid = read_info["partial_hbm_bid"]
                if partial_bid is not None:
                    hbm_k, hbm_v = state.hbm_kv[layer_name]
                    main_k_sum += hbm_k[partial_bid].float().sum().item()
                    main_v_sum += hbm_v[partial_bid].float().sum().item()
                indexer_tensor = read_info["indexer_tensor"]
                indexer_ids = read_info["d_indexer_ids"]
                if indexer_tensor is not None and indexer_ids:
                    scale = (
                        indexer_tensor.shape[0]
                        // state.num_manager_blocks
                    )
                    kernel_ids = [
                        block_id * scale + offset
                        for block_id in indexer_ids
                        for offset in range(scale)
                    ]
                    indexer_sum = (
                        indexer_tensor[kernel_ids].float().sum().item()
                    )
                else:
                    indexer_sum = 0.0
                logger.info(
                    "MFV D layer %s req %s main_k=%.6f main_v=%.6f "
                    "idx_post=%.6f",
                    layer_name,
                    read_info["ext_req_id"],
                    main_k_sum,
                    main_v_sum,
                    indexer_sum,
                )
            except Exception as error:
                logger.warning(
                    "MFV D checksum failed for %s: %s", layer_name, error
                )
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MembPull D finished read: layer=%s, req=%s, main_pages=%d, "
                "indexer_pages=%d, transfers=%d",
                layer_name,
                read_info["ext_req_id"],
                read_info["n_main"],
                read_info["n_indexer"],
                read_info["num_transfers"],
            )

    def _do_read_batch(
        self,
        layer_name: str,
        read_reqs: list[tuple[str, list[list[int]]]],
    ) -> None:
        if self._p_session is None:
            raise RuntimeError("MF_META not received before READ_READY_BATCH")
        layer = self._resolve_read_layer(layer_name)
        want_info = bool(
            envs.VLLM_ASCEND_MF_VERIFY or envs.VLLM_ASCEND_SFA_DEBUG
        )
        all_local_ptrs: list[int] = []
        all_peer_ptrs: list[int] = []
        all_lengths: list[int] = []
        read_infos: list[dict[str, Any]] = []
        for ext_req_id, p_block_ids_by_group in read_reqs:
            local_ptrs, peer_ptrs, lengths, read_info = (
                self._build_req_descriptors(
                    layer,
                    ext_req_id,
                    p_block_ids_by_group,
                    want_info,
                )
            )
            all_local_ptrs.extend(local_ptrs)
            all_peer_ptrs.extend(peer_ptrs)
            all_lengths.extend(lengths)
            if read_info is not None:
                read_infos.append(read_info)
        if not all_local_ptrs:
            raise RuntimeError(
                f"No SFAPD transfer descriptors for layer {layer_name}"
            )
        ret = self.engine.batch_transfer_sync_read(
            self._p_session,
            all_local_ptrs,
            all_peer_ptrs,
            all_lengths,
        )
        if ret != 0:
            raise RuntimeError(
                f"memfabric batch read failed for layer {layer_name}, ret={ret}"
            )
        for read_info in read_infos:
            self._log_read_result(read_info)
