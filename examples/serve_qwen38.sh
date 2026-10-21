#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# [Role] Launch the documented single-node A3 W8A8 Qwen3.8 service.
# [Invariants] TP4/DP4 needs 16 visible devices; requires Model Runner V1.
# [Timeouts & Locks] Foreground service; no locks or launch timeout.
# [Side-Effect Policy] Loads model weights onto NPU and listens on port 8088.
# Usage: bash examples/serve_qwen38.sh /models/Qwen3.8-Flash-Next-w8a8-mtp [vllm options...]
# Reference: docs/source/tutorials/models/Qwen3.8-Flash-Next.md, section 5.1.
set -e

model_path="${1:?Usage: $0 MODEL_PATH [vllm options...]}"
shift
unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
# shellcheck source=/dev/null
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# shellcheck source=/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh
set -uo pipefail

export VLLM_USE_V2_MODEL_RUNNER=0
exec vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port 8088 \
    --served-model-name qwen3.8-flash-next \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 135168 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.93 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
    "$@"
