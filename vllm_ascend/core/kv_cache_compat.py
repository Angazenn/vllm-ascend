# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Legacy cache types used by Ascend while moving to vLLM's manager configs."""

from dataclasses import dataclass

from vllm.v1 import kv_cache_interface as kvi

USES_MANAGER_CONFIGS = hasattr(kvi, "KVCacheLayerConfig")

if not USES_MANAGER_CONFIGS:
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec as CircularBufferSpec,
    )
    from vllm.v1.kv_cache_interface import (
        CrossAttentionSpec as CrossAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        EncoderOnlyAttentionSpec as EncoderOnlyAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec as FullAttentionSpec,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec as KVCacheGroupSpec,
    )
    from vllm.v1.kv_cache_interface import (
        RSWASpec as RSWASpec,
    )
    from vllm.v1.kv_cache_interface import (
        SlidingWindowMLASpec as SlidingWindowMLASpec,
    )
    from vllm.v1.kv_cache_interface import (
        SlidingWindowSpec as SlidingWindowSpec,
    )
    from vllm.v1.kv_cache_interface import (
        UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
    )
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry as KVCacheSpecRegistry
else:

    @dataclass(frozen=True, kw_only=True)
    class FullAttentionSpec(kvi.AttentionSpec):
        sliding_window: int | None = None
        attention_chunk_size: int | None = None

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowSpec(kvi.AttentionSpec):
        sliding_window: int
        extra_retained_tokens: int = 0
        bounded_replay: bool = False

        @property
        def num_retained_tokens(self) -> int:
            return self.sliding_window + self.extra_retained_tokens

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowMLASpec(SlidingWindowSpec):
        head_size_v: int = 0
        cache_dtype_str: str | None = None
        alignment: int | None = None
        model_version: str | None = None

    @dataclass(frozen=True, kw_only=True)
    class CircularBufferSpec(kvi.AttentionSpec):
        @property
        def uses_slot_mapping(self) -> bool:
            return False

        @property
        def prefix_cacheable(self) -> bool:
            return False

    @dataclass(frozen=True, kw_only=True)
    class CrossAttentionSpec(kvi.AttentionSpec):
        pass

    @dataclass(frozen=True, kw_only=True)
    class EncoderOnlyAttentionSpec(kvi.AttentionSpec):
        pass

    @dataclass(frozen=True, kw_only=True)
    class RSWASpec(FullAttentionSpec):
        rswa_window: int

    @dataclass(frozen=True)
    class UniformTypeKVCacheSpecs(kvi.KVCacheSpec):
        kv_cache_specs: dict[str, kvi.KVCacheSpec]

        @property
        def first_spec(self) -> kvi.KVCacheSpec:
            return next(iter(self.kv_cache_specs.values()))

        @property
        def page_size_bytes(self) -> int:
            return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

        @classmethod
        def from_specs(cls, specs: dict[str, kvi.KVCacheSpec]):
            if not specs:
                return None
            first = next(iter(specs.values()))
            if any(
                (spec.block_size, spec.dcp_sharded) != (first.block_size, first.dcp_sharded) for spec in specs.values()
            ):
                return None
            return cls(first.block_size, specs, dcp_sharded=first.dcp_sharded)

    @dataclass
    class KVCacheGroupSpec:
        layer_names: list[str]
        kv_cache_spec: kvi.KVCacheSpec
        is_eagle_group: bool = False
        host_resident: bool = False
        enable_kv_transfer: bool = True
        role: kvi.KVCacheGroupRole = kvi.KVCacheGroupRole.DEFAULT

    class KVCacheSpecRegistry:
        """Keep registrations local to Ascend's legacy spec consumers."""

        _managers: dict[type, type] = {}
        _uniform_types: dict[type, type] = {}

        @classmethod
        def register(cls, kvcache_spec_cls, manager_class, uniform_type_base_spec=None):
            cls._managers[kvcache_spec_cls] = manager_class
            cls._uniform_types[kvcache_spec_cls] = uniform_type_base_spec or kvcache_spec_cls

        @classmethod
        def get_manager_class(cls, spec):
            return cls._managers[type(spec)]

        @classmethod
        def get_uniform_type_base_spec(cls, spec):
            return cls._uniform_types[type(spec)]
