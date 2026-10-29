# SPDX-License-Identifier: Apache-2.0
"""[Role] Construct kernel-facing views over existing Ascend cache storage.

[Invariants] Preserve component packing, storage offsets, and page padding.
[Timeouts & Locks] No waits or locks; called during cache initialization.
[Side-Effect Policy] Return zero-copy views; never allocate or write cache data.
"""

from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def uses_qwen3_5_cache_views(vllm_config: "VllmConfig") -> bool:
    """Isolate the new view interface to Qwen3.5 dense and MoE text layers."""
    text_config = getattr(getattr(vllm_config, "model_config", None), "hf_text_config", None)
    return getattr(text_config, "model_type", None) in (
        "qwen3_5_text",
        "qwen3_5_moe_text",
    )


def attention_cache_views(
    raw: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    page_stride_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve V2's [page: K | V | pad] packing in zero-copy component views."""
    dtype_size = dtype.itemsize
    component_page_elements = prod(shape[2:])
    if page_stride_bytes % dtype_size or page_stride_bytes < 2 * component_page_elements * dtype_size:
        raise ValueError("Attention kernel pages must fit both dtype-aligned K/V components.")
    strides = tuple(prod(shape[dim + 1 :]) for dim in range(2, len(shape)))
    combined = torch.as_strided(
        raw.view(dtype),
        size=shape,
        stride=(component_page_elements, page_stride_bytes // dtype_size, *strides),
        storage_offset=raw.storage_offset() * raw.element_size() // dtype_size,
    )
    return combined[0], combined[1]


def mamba_cache_views(
    raw: torch.Tensor,
    shapes: Sequence[tuple[int, ...]],
    dtypes: Sequence[torch.dtype],
    page_size_bytes: int,
) -> list[torch.Tensor]:
    """Keep conv and recurrent state in successive regions of each page."""
    if raw.nbytes % page_size_bytes:
        raise ValueError("Mamba cache allocation is not a whole number of physical pages.")
    num_blocks = raw.nbytes // page_size_bytes
    component_offset = raw.storage_offset() * raw.element_size()
    views = []
    for shape, dtype in zip(shapes, dtypes):
        dtype_size = dtype.itemsize
        strides = tuple(prod(shape[dim + 1 :]) for dim in range(len(shape)))
        views.append(
            torch.as_strided(
                raw.view(dtype),
                size=(num_blocks, *shape),
                stride=(page_size_bytes // dtype_size, *strides),
                storage_offset=component_offset // dtype_size,
            )
        )
        component_offset += prod(shape) * dtype_size
    return views
