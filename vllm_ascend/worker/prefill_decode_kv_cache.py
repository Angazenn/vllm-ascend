# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrefillDecodeKVCache:
    """Composite per-layer cache for prefill scratch plus decode storage."""

    prefill: Any
    decode: Any


def is_prefill_decode_kv_cache(kv_cache: Any) -> bool:
    return isinstance(kv_cache, PrefillDecodeKVCache)


def get_prefill_kv_cache(kv_cache: Any) -> Any:
    if isinstance(kv_cache, PrefillDecodeKVCache):
        return kv_cache.prefill
    return kv_cache


def get_decode_kv_cache(kv_cache: Any) -> Any:
    if isinstance(kv_cache, PrefillDecodeKVCache):
        return kv_cache.decode
    return kv_cache


def _metadata_has_prefill(attn_metadata: Any) -> bool:
    is_prefilling = getattr(attn_metadata, "is_prefilling", None)
    if is_prefilling is not None:
        if isinstance(is_prefilling, torch.Tensor):
            return bool(is_prefilling.any().item())
        return any(bool(value) for value in is_prefilling)

    num_prefills = getattr(attn_metadata, "num_prefills", None)
    if num_prefills is not None:
        return int(num_prefills) > 0

    return False


def select_kv_cache_for_metadata(kv_cache: Any, attn_metadata: Any) -> Any:
    if not isinstance(kv_cache, PrefillDecodeKVCache):
        return kv_cache
    if attn_metadata is None or _metadata_has_prefill(attn_metadata):
        return kv_cache.prefill
    return kv_cache.decode


def select_mirror_decode_kv_cache(kv_cache: Any, attn_metadata: Any) -> Any | None:
    if not isinstance(kv_cache, PrefillDecodeKVCache):
        return None
    if attn_metadata is None or not _metadata_has_prefill(attn_metadata):
        return None
    return kv_cache.decode
