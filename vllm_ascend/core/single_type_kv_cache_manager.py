# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend allocation notifications for circular state sharing physical pages."""

from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager


class AscendCircularBufferManager(CircularBufferManager):
    # Ring allocations can reuse pages previously owned by attention groups.
    # Report those IDs to the worker so it zeros the shared backing.
    zeroes_new_blocks = True
