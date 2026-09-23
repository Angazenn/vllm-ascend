# Sparse offload kernel provenance

The FP16/BF16 kernels are imported directly from
[vLLM-Ascend PR #16640](https://github.com/vllm-project/vllm-ascend/pull/16640),
revision `a9823977149172f1604d9f2a1937224d0b11646e`:

- `fused_lightning_indexer_manage`
- `fused_scatter_copy_sparse_flash_attention`

Their build definitions, public ACLNN adapters and standalone tests are carried
with the implementations. Serving and Meta dispatch use the PR's public operator
names. The former `fused_li_manage_mtp`, `fused_copy_sfa_mtp` and standalone
`first_fill_scatter_copy` implementations are replaced.

## Local device-check adaptation

The only source adaptation relative to the PR is in the copy-SFA Torch adapter:
it permits contiguous CPU views for `dram_k_rope` and `dram_kv_cache`, backed by
registered MemFabric memory. Other device inputs remain on the query NPU. This
device check does not establish registration; the existing offload manager owns
allocation and registration. Arbitrary CPU allocations are not valid DRAM source
buffers. No cache is moved to NPU.

Both adapters use the PR's `EXEC_NPU_CMD` without local tensor keepalive tuples.
Copy-SFA has one launch: its internal first-fill stage performs copying,
synchronizes all MIX tasks, and then executes attention. There is no separate
first-fill ACLNN launch.

Both `op_kernel` implementations match the PR exactly. The previous local LIM
short-row index invalidation is omitted: short requests use Copy-SFA with
`num_cache_tokens=0`, which reads the causal KV range directly without consuming
TopK indices. Inactive rows have zero logical length. LIM's unused short-row
TopK entries are therefore not required to match the non-offload indexer's
per-query `-1` padding.

The source-aware gather read-ahead, physical-address reuse, paired cache
writeback and generalized LIM miss-union optimization come from the PR. Their
buffer layouts and synchronization are retained together. Existing framework
metadata, exact device sequence lengths, inactive-row handling, MTP index reuse,
and tail restoration retain their current behavior.

## Other operator family

The independent `fused_li_manage_mtp_c8` remains unchanged. It originated in the
previous nano integration through `0edbec116f23aa8f343553ee753a3699649308cf`
(initial import `11fbfda30`) and is not supplied by PR #16640. It is built for A3
but is not selected by the current serving configuration.

The replaced FP16/BF16 implementations had been pinned to nano LIM
`012962af05f06bf1bdd089ca7e7e4357d021682f` and copy-SFA
`1518a90dd17592dc3aa96c16869cfde597a250b4`.

## Validation scope

The PR's standalone operator suites cover its native contracts. Additional
`test_nano_lim_integration.py` and `test_nano_copy_sfa_integration.py` regressions
cover inactive-row isolation, serving head adaptation, registered host-memory
sources, real misses, and graph replay. Run registered-DRAM cases in an isolated
operator-test process with the serving-compatible MemFabric runtime; they own a
single-rank pool and release it after NPU synchronization.

New native binaries must be built in a clean candidate build directory. Previous
kernel and model test results do not validate these replacements. Validate both
outputs and cache contents before PD correctness and performance comparisons.
