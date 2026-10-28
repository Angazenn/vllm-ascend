"""Upstream view and binding contracts for one shared cache allocation."""

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_views import allocate_shared_cache, create_shared_cache_views


def test_disjoint_dense_banks_and_mixed_dtype_state_bind_without_copy():
    state = MambaSpec(block_size=128, shapes=((2, 3), (4,)), dtypes=(torch.bfloat16, torch.float32))
    ple = MambaSpec(block_size=128, shapes=((5,),), dtypes=(torch.bfloat16,))
    hidden = HiddenStateCacheSpec(block_size=128, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    descriptors = [
        KVCacheTensor(size=2560, layers=["state"], offset=16, layer_stride=128, block_stride=64),
        KVCacheTensor(size=2560, layers=["ple"], offset=256, layer_stride=30, block_stride=10),
        KVCacheTensor(size=2560, layers=["hidden"], offset=512, layer_stride=2048, block_stride=1024),
    ]
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=descriptors,
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in (("state", state), ("ple", ple), ("hidden", hidden))
        ],
    )
    raw = allocate_shared_cache(config, lambda size: torch.zeros(size, dtype=torch.int8))
    assert raw["state"] is raw["ple"] is raw["hidden"]
    views = create_shared_cache_views(config, raw, KVCacheLayout.BLHNC, [128, 128, 128])
    bound = {}
    for name, spec in (("state", state), ("ple", ple)):
        layer = SimpleNamespace(
            get_state_shape=lambda spec=spec: spec.shapes, get_state_dtype=lambda spec=spec: spec.dtypes
        )
        MambaBase.bind_kv_cache(layer, views[name])
        bound[name] = layer.kv_cache
    conv, ssm = bound["state"]
    assert conv.data_ptr() - raw["state"].data_ptr() == 16
    assert ssm.data_ptr() - raw["state"].data_ptr() == 28
    assert ssm.stride(0) * ssm.element_size() == 64
    assert bound["ple"][0].is_contiguous()
    assert views["hidden"].is_contiguous()
    snapshots = (conv.clone(), ssm.clone(), views["hidden"].clone())
    bound["ple"][0][torch.tensor([1])] = 7
    for actual, expected in zip((conv, ssm, views["hidden"]), snapshots, strict=True):
        torch.testing.assert_close(actual, expected)
    views["hidden"].fill_(9)
    torch.testing.assert_close(bound["ple"][0][1], torch.full((5,), 7, dtype=torch.bfloat16))
    assert torch.count_nonzero(raw["state"][:16]) == 0
