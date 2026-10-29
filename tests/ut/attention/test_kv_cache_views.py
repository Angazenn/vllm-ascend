# SPDX-License-Identifier: Apache-2.0
"""CPU storage tests for backend-owned Qwen3.5 cache views."""

import importlib.util
import sys
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
sys.modules[module_spec.name] = cache_views
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


def test_explicit_attention_placement_keeps_pool_capacity_and_gaps():
    backing = torch.zeros(768, dtype=torch.int8)
    raw = backing[64:]
    # Only two blocks belong to this layer, despite the larger shared pool.
    key, value = attention_cache_views(raw, (2, 2, 2, 1, 4), torch.float16, 128, offset_bytes=32)
    key[1].fill_(3)
    value[0].fill_(5)
    expected = torch.zeros_like(backing)
    expected[64 + 32 + 128 : 64 + 32 + 128 + 16].view(torch.float16).fill_(3)
    expected[64 + 32 + 16 : 64 + 32 + 32].view(torch.float16).fill_(5)
    torch.testing.assert_close(backing, expected)
    assert key.shape[0] == 2


def test_explicit_mamba_placement_preserves_block_stride_and_storage_offset():
    backing = torch.zeros(768, dtype=torch.int8)
    raw = backing[64:]
    placement = cache_views.CacheViewPlacement(num_blocks=2, offset_bytes=32, block_stride_bytes=256)
    conv, ssm = mamba_cache_views(raw, ((4,), (2,)), (torch.float16, torch.float32), 128, placement=placement)
    conv[1].fill_(2)
    ssm[0].fill_(4)
    expected = torch.zeros_like(backing)
    expected[64 + 32 + 256 : 64 + 32 + 256 + 8].view(torch.float16).fill_(2)
    expected[64 + 32 + 8 : 64 + 32 + 16].view(torch.float32).fill_(4)
    torch.testing.assert_close(backing, expected)
    assert conv.shape == (2, 4)
    assert ssm.shape == (2, 2)


@pytest.mark.parametrize("layer_stride", [0, 384])
def test_hybrid_plan_materializes_one_pool_with_descriptor_aliases(monkeypatch, layer_stride):
    """Planning preserves physical overlays and zero-stride layerwise reuse."""
    from types import SimpleNamespace

    # The planner needs only descriptor placement and spec page bytes, so it
    # can be exercised without initializing vLLM or the NPU plugin.
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.kv_cache_views", cache_views)
    plan_spec = importlib.util.spec_from_file_location(
        "ascend_cache_plan", Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/kv_cache_plan.py"
    )
    planner = importlib.util.module_from_spec(plan_spec)
    monkeypatch.setitem(sys.modules, plan_spec.name, planner)
    plan_spec.loader.exec_module(planner)
    page_bytes, blocks, offset, pool_size = 128, 3, 32, 800
    descriptors = [
        SimpleNamespace(
            size=pool_size,
            layers=["attention.0", "attention.2"],
            offset=offset,
            layer_stride=layer_stride,
            block_stride=page_bytes,
            host_resident=False,
        ),
        SimpleNamespace(
            size=pool_size,
            layers=["gdn.1"],
            offset=offset,
            layer_stride=384,
            block_stride=page_bytes,
            host_resident=False,
        ),
    ]
    config = SimpleNamespace(kv_cache_tensors=descriptors, num_blocks_of=lambda _descriptor: blocks)
    specs = {name: SimpleNamespace(page_size_bytes=page_bytes) for name in ("attention.0", "attention.2", "gdn.1")}
    plan = planner.build_qwen3_5_cache_plan(config, specs, {"attention.4": "attention.0"})
    requested_sizes = []

    def allocate(size):
        requested_sizes.append(size)
        return torch.zeros(size + 64, dtype=torch.int8)[64:]

    allocations = planner.allocate_cache_storage(plan, allocate)
    assert requested_sizes == [pool_size]
    assert plan.layers["attention.2"].placement.offset_bytes == offset + layer_stride
    assert plan.layers["gdn.1"].placement.num_blocks == blocks
    raw = allocations[plan.layers["gdn.1"].allocation_index]
    conv, ssm = mamba_cache_views(
        raw,
        ((4,), (2,)),
        (torch.float16, torch.float32),
        page_bytes,
        placement=plan.layers["gdn.1"].placement,
    )
    conv[1].fill_(2)
    ssm[2].fill_(7)
    expected = torch.zeros(pool_size, dtype=torch.int8)
    expected[offset + page_bytes : offset + page_bytes + 8].view(torch.float16).fill_(2)
    expected[offset + 2 * page_bytes + 8 : offset + 2 * page_bytes + 16].view(torch.float32).fill_(7)
    torch.testing.assert_close(raw, expected)
