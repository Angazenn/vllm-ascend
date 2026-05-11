# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Thoroughly review [AGENTS.md](AGENTS.md) before beginning any work — it contains full contributor guidelines.

## Build and Development

```bash
# Install in editable mode (requires NPU SDK: CANN 8.5.0, torch 2.9.0, torch-npu 2.9.0)
SOC_VERSION=<chip> pip install -e .

# Build C++ extensions only (for dev iteration)
SOC_VERSION=<chip> COMPILE_CUSTOM_KERNELS=1 pip install -e .

# For CPU-only / UT environments (skip C++ extension build)
COMPILE_CUSTOM_KERNELS=0 pip install -e .
```

Common `SOC_VERSION` values: `ascend910b1` (Atlas A2), `ascend910_9391` (Atlas A3), `ascend310p1` (Atlas 300I Duo).

## Lint and Format

```bash
# Full pre-commit check (use this before pushing)
bash format.sh ci

# Quick ruff linting
ruff check vllm_ascend/

# Ruff auto-format
ruff format vllm_ascend/
```

## Testing

```bash
# Run a specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run a specific test function
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run unit tests for a module
pytest -sv tests/ut/batch_invariant/

# Run NPU-specific e2e tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/
```

## Architecture

vLLM Ascend is a **hardware plugin** for running vLLM on Huawei Ascend NPUs. It integrates via vLLM's pluggable hardware interface — it does NOT fork vLLM but extends it at runtime.

### Entry Point

`vllm_ascend/__init__.py` registers 5 plugin entry points:
- **platform** (`NPUPlatform`) — the core hardware abstraction
- **kv_connector** — KV cache transfer for P/D disaggregation
- **model_loader** — NetLoader and RForkLoader for weight loading
- **service_profiling** — profiling config

### Key Directories

| Directory | Purpose |
|---|---|
| `vllm_ascend/platform.py` | `NPUPlatform` class — vLLM hardware plugin interface. Configures compilation mode (piecewise/full/NONE), scheduler, parallel config, attention backends, and NPU-specific defaults |
| `vllm_ascend/worker/` | V1 model runner (`model_runner_v1.py`) and NPU worker (`worker.py`) |
| `vllm_ascend/worker/v2/` | V2 model runner — used when `VLLM_USE_V2_MODEL_RUNNER=1` |
| `vllm_ascend/patch/` | Runtime monkey-patching of upstream vLLM. `patch/platform/` for distributed/scheduling patches; `patch/worker/` for model-specific patches (DeepSeek, Qwen3, etc.) |
| `vllm_ascend/ops/` | NPU-specific custom ops (linear, layernorm, rotary embedding, MLA, fused MoE) |
| `vllm_ascend/compilation/` | ACL graph capture (`acl_graph.py`), custom compiler backend, and inductor graph fusion passes |
| `vllm_ascend/attention/` | Attention backends: standard, MLA (Multi-head Latent Attention), and SFA (Sparse Flash Attention) |
| `vllm_ascend/_310p/` | Ascend 310P-specific variant (model runner, attention, ops, quantization) |
| `vllm_ascend/distributed/` | HCCL communicator, KV transfer for P/D disaggregation, parallel state utilities |
| `vllm_ascend/quantization/` | Ascend quantization (W8A8, Compressed Tensors, ModelSlim) |
| `vllm_ascend/spec_decode/` | Speculative decoding proposers (Eagle, Medusa, n-gram, suffix) |
| `csrc/` | C++ CUDA-like kernels compiled via CMake (MoE routing, Fused MC2, attention, LoRA) |

### Key Patterns

- **Patching** (`patch/`): Runtime monkey-patching of upstream vLLM classes/methods. Use `patch/worker/` for model-specific NPU adaptations and `patch/platform/` for infrastructure-level changes. New patches require strict architectural review.

- **Dual model runner versions**: Both V1 (`model_runner_v1.py`) and V2 (`worker/v2/model_runner.py`) are maintained. V2 is used when `VLLM_USE_V2_MODEL_RUNNER=1`.

- **Environment variables**: All must be centralized in `vllm_ascend/envs.py` using `VLLM_ASCEND_*` naming. Never hardcode env var names throughout the codebase.

- **Compilation modes**: `PIECEWISE` (ACL graph per-layer capture, default for NPU), `FULL_DECODE_ONLY` (experimental full graph), `NONE` (eager). Piecewise is enforced as default over FULL_AND_PIECEWISE.

- **`tensor.item()` is expensive on NPU**: Forces CPU-NPU sync. Avoid in hot paths; prefer device-side operations.
