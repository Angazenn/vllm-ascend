# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager, OffloadMLAAttentionManager
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def is_direct_sfa_kv_offload(vllm_config: VllmConfig) -> bool:
    """Whether the direct layerwise SFA decode-offload layout is active."""
    transfer_config = vllm_config.kv_transfer_config
    additional_config = vllm_config.additional_config or {}
    return bool(
        additional_config.get("use_offload", False)
        and transfer_config is not None
        and transfer_config.kv_connector == "SFAKVOffloadConnector"
        and transfer_config.kv_connector_extra_config.get(
            "use_layerwise", False
        )
    )


def get_sfa_layerwise_ascend_store_config(
    vllm_config: VllmConfig,
) -> dict[str, Any] | None:
    """Return the effective layerwise AscendStore configuration.

    PD prefill uses AscendStore as a child of MultiConnector. Layerwise pool
    sizing options may be placed on the parent for compatibility with the
    model-runner memory calculation, so merge those defaults into the child
    configuration before returning it.
    """
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None:
        return None
    parent_extra = transfer_config.kv_connector_extra_config or {}

    if transfer_config.kv_connector == "AscendStoreConnector":
        return dict(parent_extra) if parent_extra.get("use_layerwise", False) else None
    if transfer_config.kv_connector != "MultiConnector":
        return None

    parent_layerwise = {
        key: value
        for key, value in parent_extra.items()
        if key.startswith("layerwise_") or key == "use_layerwise"
    }
    for child in parent_extra.get("connectors", []):
        if child.get("kv_connector") != "AscendStoreConnector":
            continue
        effective = dict(parent_layerwise)
        effective.update(child.get("kv_connector_extra_config") or {})
        if effective.get("use_layerwise", False):
            return effective
    return None


def uses_split_sfa_decode_offload_layout(vllm_config: VllmConfig) -> bool:
    """Whether the worker needs split main/indexer decode-offload caches."""
    if not (vllm_config.additional_config or {}).get("use_offload", False):
        return False
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None:
        return False
    if not (transfer_config.kv_connector_extra_config or {}).get(
        "use_layerwise", False
    ):
        return False
    if transfer_config.kv_connector == "SFAKVOffloadConnector":
        return True
    return bool(
        transfer_config.kv_connector == "SFAPDCpuOffloadConnector"
        and transfer_config.is_kv_consumer
    )


def _get_c8_k_cache_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8


def _get_c8_k_scale_cache_dtype() -> torch.dtype:
    return torch.float32 if get_ascend_device_type() == AscendDeviceType.A5 else torch.float16


def offload_indexer_pad_dim(
    index_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
) -> int:
    return index_head_dim * qk_rope_head_dim // kv_lora_rank


def offload_main_kv_head_dims_for_pool_split(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> list[int]:
    return [kv_lora_rank, qk_rope_head_dim]


def offload_indexer_kernel_block_size(
    mla_block_size: int,
    kv_lora_rank: int,
    index_head_dim: int,
) -> int:
    return mla_block_size * kv_lora_rank // index_head_dim


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    # Legacy offload indexer specs still use this field until that path adopts
    # a dedicated split-cache specification.
    sparse_head_dim: tuple[int, ...] | None = None
    # Sparse C8 changes the main cache into one packed byte tensor. Keep that
    # main-cache property here; indexer-specific C8 properties belong to the
    # indexer spec.
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_cache_dtype)
    c8_k_scale_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_scale_cache_dtype)
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads

            ckv_head_dim, qk_rope_head_dim, index_head_dim = self.sparse_head_dim
            assert qk_rope_head_dim == 0

            ckv_bytes = num_heads_per_page * ckv_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            qli_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            qli_scale_bytes = (
                num_heads_per_page * self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_scale_cache_dtype)
                if index_head_dim > 0
                else 0
            )
            return ckv_bytes + qli_bytes + qli_scale_bytes
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        layout_set = {
            (
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                spec.scale_dim,
                spec.scale_dtype,
                spec.sparse_head_dim,
                spec.dtype,
            )
            for spec in specs
        }
        assert len(layout_set) == 1, (
            "All attention layers in the same KV cache group must use the same KV cache layout."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        assert len(cache_sparse_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse C8 setting."
        )
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert len(sfa_dcp_replicated_indexer_size_set) == 1, (
            "All attention layers in the same KV cache group must use the same SFA DCP replicated indexer size."
        )

        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            scale_dim=specs[0].scale_dim,
            scale_dtype=specs[0].scale_dtype,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            cache_sparse_c8=specs[0].cache_sparse_c8,
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(FullAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The scheduler should treat this as a full-attention-compatible cache so it
    can share block ids with the MLA cache in the same UniformType group. The
    model runner still allocates it as an independent physical cache tensor.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_c8: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return (
            self.sfa_dcp_replicated_indexer_size
            * num_heads_per_page
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSFAIndexerCacheSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        dtype_set = set(spec.dtype for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(dtype_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(cache_sparse_c8_set) == 1
            and len(sfa_dcp_replicated_indexer_size_set) == 1
        ), (
            "All SFA indexer cache layers in the same KV cache group must use "
            "the same dtype, scale layout, quantization method, sparse C8 "
            "setting and DCP replication size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=dtype_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            cache_sparse_c8=cache_sparse_c8_set.pop(),
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendSFALayerwiseIndexerCacheSpec(FullAttentionSpec):
    """BF16 SFA indexer cache sharing layerwise AscendStore raw pools.

    ``head_size`` is the semantic indexer width. ``indexer_pad_dim`` reserves
    the remainder of the physical page so vLLM can unify this spec with the
    main MLA page while the kernel still sees 128-token virtual blocks.
    """

    indexer_pad_dim: int = 0
    cache_dtype_str: str | None = None

    @property
    def page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.indexer_pad_dim)
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert specs and all(isinstance(spec, cls) for spec in specs)
        layout = {
            (
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                spec.dtype,
                spec.indexer_pad_dim,
                spec.cache_dtype_str,
            )
            for spec in specs
        }
        assert len(layout) == 1, (
            "All layerwise SFA indexers in one group must use the same "
            "physical cache layout."
        )
        return specs[0]


@dataclass(frozen=True, kw_only=True)
class AscendSFAOffloadIndexerCacheSpec(FullAttentionSpec):
    """BF16 resident indexer cache for direct SFA decode offload.

    The semantic indexer width is ``head_size``. ``indexer_pad_dim`` makes
    one unified manager page match the main MLA page, allowing the indexer K
    view to alias the main pool's raw K storage without another allocation.
    """

    indexer_pad_dim: int = 0
    cache_dtype_str: str | None = None

    @property
    def page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.indexer_pad_dim)
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert specs and all(isinstance(spec, cls) for spec in specs)
        layout = {
            (
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                spec.dtype,
                spec.indexer_pad_dim,
                spec.cache_dtype_str,
            )
            for spec in specs
        }
        assert len(layout) == 1, (
            "All direct SFA offload indexers in one group must use the same "
            "physical cache layout."
        )
        return specs[0]


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        pass

    @property
    def storage_block_size(self) -> int:
        return self.block_size

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class OffloadMLAAttentionSpec(AttentionSpec):
    @property
    def page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """The maximum possible memory usage of this KV cache in bytes.

        Feeds the startup must-fit check (_check_enough_kv_cache_memory) and the
        max_model_len auto-fit -- NOT num_blocks pool sizing, so changing this
        only relaxes the max_model_len limit, it does not shrink the pool.
        """
        ktc = vllm_config.kv_transfer_config
        if ktc is not None and ktc.kv_role == "kv_consumer":
            # PD D-side keeps only the MLA resident tail in HBM: the
            # remote-prefill prefix is null-padded, and completed decode blocks
            # are freed in-place after save. Reserve current partial + one
            # handoff block per request for the must-fit check.
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            blocks_per_req = 2
            return blocks_per_req * max_num_seqs * self.page_size_bytes
        # Producer (P) or local offload (kv_both) or no kv-transfer: the prefix
        # transits HBM during prefill before being offloaded, so the peak is the
        # full max_model_len.
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


def make_offload_main_mla_spec(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> OffloadMLAAttentionSpec:
    return OffloadMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )


def make_offload_indexer_mla_spec(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    cache_dtype_str: str,
    index_head_dim: int,
    indexer_pad_dim: int | None = None,
    sparse_head_dim: tuple[int, ...] | None = None,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
) -> AscendMLAAttentionSpec:
    """Build the legacy BF16 indexer offload spec."""
    if indexer_pad_dim is None:
        assert kv_lora_rank is not None and qk_rope_head_dim is not None
        indexer_pad_dim = offload_indexer_pad_dim(
            index_head_dim, qk_rope_head_dim, kv_lora_rank
        )
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        cache_dtype_str=cache_dtype_str,
        sparse_head_dim=sparse_head_dim,
        scale_dim=0,
        cache_sparse_c8=False,
    )


def make_sfa_offload_indexer_spec(
    *,
    block_size: int,
    num_kv_heads: int,
    index_head_dim: int,
    indexer_pad_dim: int,
    dtype: torch.dtype,
    cache_dtype_str: str,
) -> AscendSFAOffloadIndexerCacheSpec:
    """Build the real-indexer spec for direct BF16 decode offload."""
    return AscendSFAOffloadIndexerCacheSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=index_head_dim,
        indexer_pad_dim=indexer_pad_dim,
        dtype=dtype,
        cache_dtype_str=cache_dtype_str,
    )


def register_ascend_kv_cache_specs() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendMLAAttentionSpec,
        manager_class=CompressAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFAIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFALayerwiseIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFAOffloadIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSlidingWindowMLASpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )
    KVCacheSpecRegistry.register(
        OffloadMLAAttentionSpec,
        OffloadMLAAttentionManager,
        uniform_type_base_spec=OffloadMLAAttentionSpec,
    )
