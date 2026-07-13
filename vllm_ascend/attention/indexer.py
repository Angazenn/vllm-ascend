from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class AscendSFAIndexerBackend(AttentionBackend):
    """Placeholder backend for split SFA indexer cache layers.

    The SFA indexer cache is represented as its own AttentionLayerBase so the
    KV-cache planner can assign an independent physical tensor while sharing
    block ids with the main MLA cache group. The current SFA forward path still
    consumes metadata from the real ``*.attn`` layer and recomposes the legacy
    cache tuple before calling the kernel, so this backend only needs to make
    the indexer cache visible to cache initialization.

    Do not reuse AscendSFAMetadataBuilder here. It inherits vLLM's
    MLACommonMetadataBuilder, whose initializer assumes layer_names[0] points to
    a real MLAAttention object with ``prefill_backend`` in static_forward_context.
    The indexer cache layer points to DeepseekV32IndexerCache instead, which has
    no ``prefill_backend``. Keeping a cache-only builder avoids that false
    attention-layer assumption and avoids building unused indexer metadata.
    """

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFAIndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendSFAIndexerMetadataBuilder(AttentionMetadataBuilder[Any]):
    """Cache-only metadata builder for split SFA indexer cache layers."""

    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> None:
        return None


@dataclass
class AscendSFALayerwiseIndexerMetadata(AttentionMetadata):
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor


class AscendSFALayerwiseIndexerMetadataBuilder(
    AscendSFAIndexerMetadataBuilder
):
    """Expose addressing from the indexer's own layerwise cache group."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFALayerwiseIndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = getattr(
            common_attn_metadata,
            "num_input_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        return AscendSFALayerwiseIndexerMetadata(
            block_table_tensor=common_attn_metadata.block_table_tensor[:num_reqs],
            slot_mapping=common_attn_metadata.slot_mapping[:num_tokens],
        )


class AscendSFALayerwiseIndexerBackend(AscendSFAIndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_LAYERWISE_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFALayerwiseIndexerMetadataBuilder


@dataclass
class AscendSFAOffloadIndexerMetadata(AttentionMetadata):
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor


class AscendSFAOffloadIndexerMetadataBuilder(
    AscendSFAIndexerMetadataBuilder
):
    """Expose addressing owned by the resident decode indexer group."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAOffloadIndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = getattr(
            common_attn_metadata,
            "num_input_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        return AscendSFAOffloadIndexerMetadata(
            block_table_tensor=common_attn_metadata.block_table_tensor[
                :num_reqs
            ],
            slot_mapping=common_attn_metadata.slot_mapping[:num_tokens],
        )


class AscendSFAOffloadIndexerBackend(AscendSFAIndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_OFFLOAD_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFAOffloadIndexerMetadataBuilder
