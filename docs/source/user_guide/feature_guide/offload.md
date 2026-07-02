# SFA Prefill and Decode Offload

This guide covers the experimental SFA offload path for MLA sparse attention.
It combines:

- prefill offload through `AscendStoreConnector`, where layerwise prefill cache is stored in the CPU-side store backend.
- decode offload through `SFAKVOffloadConnector`, where full decode KV pages are saved to CPU memory and a small LRU resident top-k window is loaded back to HBM during decode attention.

## Launch

Enable sparse offload with `--additional-config` and use a `MultiConnector` with both connectors:

```bash
vllm serve /path/to/model \
  --enable-chunked-prefill \
  --additional-config '{
    "use_offload": true,
    "lru_resident_cache_config": {
      "buffer_size": 2048,
      "topk": 2048
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "prefill_hybrid_groups": true,
      "layerwise_num_shared_buffers": 2,
      "connectors": [
        {
          "kv_connector": "AscendStoreConnector",
          "kv_role": "kv_both",
          "kv_connector_extra_config": {
            "backend": "memcache",
            "use_layerwise": true,
            "mooncacke_rpc_port": "0",
            "layerwise_num_shared_buffers": 2,
            "prefill_hybrid_groups": true
          }
        },
        {
          "kv_connector": "SFAKVOffloadConnector",
          "kv_role": "kv_both",
          "kv_connector_extra_config": {
            "use_layerwise": true,
            "decode_offload_group_id": -1
          }
        }
      ]
    }
  }'
```

`prefill_hybrid_groups` and `layerwise_num_shared_buffers` are intentionally
specified at the `MultiConnector` level and again in the `AscendStoreConnector`
child. The model runner reads the top-level values for cache planning, while
the child connector reads its own values for store-side group selection.

`decode_offload_group_id` selects the real KV cache group. The default `-1`
uses the last group, which matches the current hybrid layout where group 0 is
the indexer cache and group 1 is the real KV cache.

## Cache Layout

With `prefill_hybrid_groups` and `use_offload` enabled, the model runner creates
two attention cache groups:

- group 0: indexer cache.
- group 1: real MLA KV cache, using `OffloadMLAAttentionSpec`.

The prefill side keeps layerwise-reused HBM buffers for both groups. The decode
side allocates a separate normal per-layer KV cache only for the real KV group.
Each attention layer receives a composite cache object containing both spaces:

- `prefill`: used when the batch contains prefill tokens.
- `decode`: used for decode-only batches and mirrored during chunk prefill so
  decode can continue from normal KV pages.

During a mixed decode and chunk-prefill step, prefill attention reads the prefill
cache, while decode offload reads the decode cache.

## Decode Offload Parameters

`lru_resident_cache_config` controls the decode resident HBM window:

- `buffer_size`: resident tokens per request. It must be a positive multiple of
  the model cache block size.
- `topk`: number of sparse indices considered for resident loading. It must be
  positive and no larger than `buffer_size`.

The decode connector only offloads the real KV group. Indexer cache remains
allocated on HBM for decode so top-k selection can run before sparse KV loading.

## Notes

- This path currently targets MLA sparse attention with the SFA backend.
- `SFAKVOffloadConnector` currently requires `use_layerwise=true`.
- The CPU memory used by decode offload is sized from the decode KV block count
  and model layer count. Reduce `gpu_memory_utilization` or increase available
  host memory if initialization reports insufficient CPU memory.
