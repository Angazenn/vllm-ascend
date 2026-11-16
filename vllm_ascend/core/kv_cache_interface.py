# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Self

import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)


def get_kv_cache_compression_ratio(kv_cache_spec: KVCacheSpec) -> int:
    """Return the number of logical tokens represented by one stored state."""
    return kv_cache_spec.tokens_per_state


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    # vLLM #53906 added an optional MLA storage-view override. It is not
    # Ascend's derived number of physical rows per logical block.
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        return kv_cache_spec.block_size // kv_cache_spec.tokens_per_state
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return kv_cache_spec.block_size
    return getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)


def is_circular_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    return bool(getattr(kv_cache_spec, "is_circular", False))


# The architectures bounded replay was written for. Upstream wires the feature
# into ``vllm/models/deepseek_v41/`` alone -- the DeepSeek-V4 tree never reads
# the switch -- so the architecture name is the honest test for "this model may
# replay". Both of this project's spellings for the target are listed: missing
# the one a checkpoint declares is a switch that can never be turned on.
#
# The DSpark draft is deliberately left out. Its SWA cache is a different class
# (``DeepseekV41DSparkSWACache``) that does not carry the field, so a draft
# checkpoint would pass a gate that then replays nothing.
BOUNDED_REPLAY_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",  # this repository's registry key
        "DeepseekV41ForConditionalGeneration",  # the VL spelling
    }
)


def supports_bounded_replay(model_config: Any) -> bool:
    """Whether this model is one the feature was written for."""
    architectures = getattr(model_config, "architectures", None) or ()
    return any(architecture in BOUNDED_REPLAY_ARCHITECTURES for architecture in architectures)


def resolve_bounded_replay(vllm_config: VllmConfig, cache_config: CacheConfig | None) -> bool:
    """Whether a sliding-window layer may declare a replay window.

    The switch comes from upstream (vLLM #56227) and is re-decided here
    because the layer upstream applies it in -- its own SWA attention layer --
    is the one Ascend replaces. The Ascend scheduler conditions, the ones that
    cannot rewind a prefix hit at all, are decided in ``platform.py`` before any
    layer exists: they turn the switch itself off, so this reads the result.

    It reads that switch and nothing else, deliberately: a V4.1 model that does
    not turn it off in its cache config replays.

    There is no capability guard beside them. Keeping a replayed window off the
    blocks below ``replay_start`` -- whose KV the sliding-window group does not
    hold, those positions having been retired to the null block -- is the
    operator's job now: band mode reads ``ori_topk_length`` as the per-token
    visible length and floors the window's left edge with it. See
    ``build_replay_swa_visible_lens`` in ``attention/dsa_v41.py`` for the read
    side and ``npu_sparse_flash_mla``'s band path for the clamp.
    """
    if not supports_bounded_replay(getattr(vllm_config, "model_config", None)):
        # Upstream wires the feature into the DeepSeek-V4.1 tree only -- the V4
        # tree's attention layer never reads the switch -- so a V4 model is
        # silent here too: the feature is not part of this model, and no user
        # can have turned it on for it.
        return False
    # The switch is absent from release lanes without bounded replay. PCP and
    # DP share the same cache contract; decoder batch coordination is handled
    # by DecoderReplayState rather than disabling the cache feature here.
    return bool(getattr(cache_config, "swa_bounded_replay", False))


def requires_padded_page_layout(kv_cache_specs: Iterable[KVCacheSpec]) -> bool:
    """Whether state caches advance by a padded page size shared with attention.

    Ascend mixed pools can bind one physical page to both an attention cache and a
    recurrent state cache. The attention side of such a pool is addressed by an explicit
    physical block stride (``indexes_kv_by_block_stride``) and the state caches are
    padded to that same page size, so every view over the shared page has to advance by
    the padded page size: otherwise a state update for one scheduler block ID would land
    in the pages owned by another block ID.

    The capability belongs to the pool rather than to a single spec, because hybrid
    models also pad Mamba pages through ``cache_config.mamba_page_size_padded`` while
    keeping the packed contiguous state layout, so padding alone does not require this
    layout.
    """
    specs = list(kv_cache_specs)
    state_specs = [spec for spec in specs if isinstance(spec, MambaSpec)]
    if not state_specs:
        return False
    if not any(getattr(spec, "page_size_padded", None) is not None for spec in state_specs):
        return False
    return any(getattr(spec, "indexes_kv_by_block_stride", False) for spec in specs)


def supports_component_major_mla_pd(vllm_config: VllmConfig) -> bool:
    """Whether PD transfer can preserve a component-major MLA cache layout.

    A component-major MLA page packs NoPE and RoPE into one physical slot:
    ``[all NoPE][all RoPE][padding]``.  MooncakeV2 pull connectors transfer
    that page as one raw block when the published base address, block length,
    and block stride describe the complete slot.  Other connectors still use
    the legacy separate NoPE/RoPE protocol.

    Without PD, local layout selection is unrestricted. With supported PD,
    both endpoints must agree on the page layout: raw transfers carry no
    metadata distinguishing component-major from token-fused FlashMLA pages.
    This helper identifies connector support, not cross-endpoint agreement.
    """
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return True

    # Connectors loaded through a custom module path are not identified by the
    # registry name, so conservatively fall back to the legacy layout.
    if getattr(kv_transfer_config, "kv_connector_module_path", None) is not None:
        return False

    return getattr(kv_transfer_config, "kv_connector", None) in {
        "MooncakeConnectorV2",
        "MooncakePullConnector",
    }


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    # TP ranks hold the full latent vector, including quantization scales.
    is_draft: bool = False
    non_causal_multi_token_decode: bool = False
    max_tp_shards: int | None = 1

    # Per-rank query heads used by FlashMLA layout selection. Auxiliary MLA
    # caches without query heads use the one-latent-head cache contract.
    num_heads: int = 1
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    # Sparse C8 changes the main cache into one packed byte tensor. Keep that
    # main-cache property here; indexer-specific C8 properties belong to the
    # indexer spec.
    cache_sparse_sfa_c8: bool = False
    cache_sparse_sfa_turboquant: bool = False
    store_on_host: bool = False
    # Ascend kernels consume padded pages through an explicit physical block
    # stride. vLLM main removed this field from AttentionSpec, but it remains
    # part of the Ascend runner/backend contract.
    indexes_kv_by_block_stride: bool = False

    # AttentionGroup replaces block_size with the kernel size for builders.
    # Keep the manager page size for DCP table expansion and address mapping.
    block_table_block_size: int | None = None

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(
            super().copy_with_new_block_size(block_size),
            block_table_block_size=self.block_table_block_size or self.block_size,
        )

    @property
    def uses_packed_sfa_main_cache(self) -> bool:
        """Whether the SFA main cache is stored in one packed tensor."""
        return self.cache_sparse_sfa_c8 or self.cache_sparse_sfa_turboquant

    @property
    def supports_single_raw_backing(self) -> bool:
        """Whether this geometry can be exposed from one raw MLA backing."""
        return (
            get_kv_cache_compression_ratio(self) == 1
            and self.model_version is None
            and not self.indexes_kv_by_block_stride
            and not self.cache_sparse_sfa_c8
        )

    @property
    def real_page_size_bytes(self) -> int:
        return (
            get_storage_block_size(self)
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if self.dcp_sharded and dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The scheduler should treat this as a full-attention-compatible cache so it
    can share block ids with the MLA cache in the same UniformType group. The
    indexer retains its own physical cache region, separate from the MLA cache.
    """

    # Indexer keys and their scales are replicated across TP ranks.
    is_draft: bool = False
    non_causal_multi_token_decode: bool = False
    max_tp_shards: int | None = 1
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_li_c8: bool = False
    cache_sparse_li_c4: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1

    block_table_block_size: int | None = None

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(
            super().copy_with_new_block_size(block_size),
            block_table_block_size=self.block_table_block_size or self.block_size,
        )

    @property
    def state_content_size_bytes(self) -> int:
        # The V2 layout planner derives strides from stored row bytes. Include
        # replicated K/scale regions so its descriptors cover the full page;
        # the indexer reshape still exposes component-major K/scale views.
        return self.sfa_dcp_replicated_indexer_size * (
            self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype)
        )

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return num_heads_per_page * self.state_content_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendIndexerKPoolTailSpec(AttentionSpec):
    """One fixed FP32 ``[ring_capacity, 2 * head_size]`` page per request.

    ``block_size`` is the logical pool size (``compress_ratio``), while
    ``ring_capacity`` also retains speculative rows. The dedicated manager
    owns one complete ring per request and never slides or grows it.
    """

    compress_ratio: int
    ring_capacity: int
    dcp_sharded: bool = False
    model_version: str = "glm5_next"
    cache_role: str = "indexer_tail"
    indexes_kv_by_block_stride: bool = True

    @property
    def is_circular(self) -> bool:
        return True

    @property
    def storage_block_size(self) -> int:
        return self.ring_capacity

    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        # One logical pool block owns the whole ring, including lookahead.
        return self.ring_capacity

    @property
    def unpadded_page_size_bytes(self) -> int:
        # The tail stores a fixed K/gate pair; the parent handles page padding.
        return 2 * self.ring_capacity * self.head_size * get_dtype_size(self.dtype)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype != torch.float32:
            raise ValueError(f"Indexer K-pool tail must use FP32, got {self.dtype}.")
        if self.num_kv_heads != 1 or self.head_size <= 0:
            raise ValueError("Indexer K-pool tail requires one K/gate pair with positive head size.")
        if self.compress_ratio <= 1 or self.ring_capacity < self.compress_ratio:
            raise ValueError("Tail ring capacity must be at least the compression ratio, which must exceed one.")
        if self.block_size != self.compress_ratio:
            raise ValueError("Tail logical block size must equal the compression ratio.")


@dataclass(frozen=True, kw_only=True)
class AscendCircularBufferSpec(MLAAttentionSpec):
    """Layout of the compressor ring's single physical page."""

    is_circular = True
