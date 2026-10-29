# SPDX-License-Identifier: Apache-2.0
"""CPU storage tests for backend-owned Qwen3.5 cache views."""

import importlib.util
from math import prod
from pathlib import Path

import pytest
import torch

# Load the complete CPU-only module without triggering the plugin's vLLM/NPU
# initialization. These tests exercise real PyTorch storage, with no device mocks.
module_spec = importlib.util.spec_from_file_location(
    "ascend_cache_views", Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kv_cache_views.py"
)
cache_views = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cache_views)
attention_cache_views = cache_views.attention_cache_views
mamba_cache_views = cache_views.mamba_cache_views


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("padding", [0, 32])
def test_attention_views_preserve_page_packing_and_layer_offset(head_first, padding):
    """K/V writes hit only their kernel page inside a shared layer slice."""
    blocks, heads, tokens, dim = 3, 2, 4, 8
    inner = (heads, tokens, dim) if head_first else (tokens, heads, dim)
    component_bytes = prod(inner) * torch.float16.itemsize
    kernel_page_bytes = 2 * component_bytes + padding
    num_kernel_blocks = blocks
    layer_bytes = num_kernel_blocks * kernel_page_bytes
    backing = torch.zeros(2 * layer_bytes, dtype=torch.int8)
    raw = backing[layer_bytes:]
    key, value = attention_cache_views(raw, (2, num_kernel_blocks, *inner), torch.float16, kernel_page_bytes)
    assert key.shape == value.shape == (num_kernel_blocks, *inner)
    assert key.stride(0) == value.stride(0) == kernel_page_bytes // 2
    assert key.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert value.data_ptr() - key.data_ptr() == component_bytes
    key[1].fill_(3)
    value[1].fill_(5)

    # Independent byte oracle includes the untouched layer, other pages,
    # and page padding rather than comparing two implementations of strides.
    expected = torch.zeros_like(backing)
    start = layer_bytes + kernel_page_bytes
    expected[start : start + component_bytes].view(torch.float16).fill_(3)
    expected[start + component_bytes : start + 2 * component_bytes].view(torch.float16).fill_(5)
    torch.testing.assert_close(backing, expected)


def test_mamba_views_preserve_mixed_dtypes_page_padding_and_overlay():
    """Conv/SSM updates retain page ownership when attention aliases the pool."""
    blocks, page_bytes, layer_offset = 3, 128, 256
    shapes = ((3, 4), (2, 4, 2))
    backing = torch.zeros(layer_offset + blocks * page_bytes, dtype=torch.int8)
    raw = backing[layer_offset:]
    key, _ = attention_cache_views(raw, (2, blocks, 2, 1, 8), torch.bfloat16, page_bytes)
    conv, ssm = mamba_cache_views(raw, shapes, (torch.bfloat16, torch.float32), page_bytes)
    assert conv.shape == (blocks, *shapes[0])
    assert ssm.shape == (blocks, *shapes[1])
    assert conv.stride(0) == 64 and ssm.stride(0) == 32
    assert ssm.data_ptr() - conv.data_ptr() == prod(shapes[0]) * 2
    conv[1].fill_(2)
    torch.testing.assert_close(key[1].flatten()[:12], torch.full((12,), 2, dtype=torch.bfloat16))
    ssm[2].fill_(4)
    expected = torch.zeros_like(backing)
    start = layer_offset + page_bytes
    expected[start : start + 24].view(torch.bfloat16).fill_(2)
    start = layer_offset + 2 * page_bytes + 24
    expected[start : start + 64].view(torch.float32).fill_(4)
    torch.testing.assert_close(backing, expected)
