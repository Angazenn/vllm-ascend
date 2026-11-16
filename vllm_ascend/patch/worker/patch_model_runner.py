# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project


from vllm.v1.worker.gpu import model_runner as upstream
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp as _dispatch_cg_and_sync_dp

from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


# TODO: Remove once upstream GPUModelRunner dispatches sharded PCP decode with
# rank-local request counts.
def dispatch_cg_and_sync_dp(cudagraph_manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs):
    vllm_config = getattr(cudagraph_manager, "vllm_config", None)
    if uniform_token_count and vllm_config is not None and is_pcp_decode_sharding_enabled(vllm_config):
        # num_tokens is already the largest rank-local count, and each sharded
        # decode request stays whole on one rank.
        num_reqs = num_tokens // uniform_token_count
    return _dispatch_cg_and_sync_dp(cudagraph_manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs)


upstream.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace

upstream.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
