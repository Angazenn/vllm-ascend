# SPDX-License-Identifier: Apache-2.0
# [Role] Configure AISBench requests for the Qwen3.8 GPQA evaluation.
# [Invariants] Batch 64; max_tokens=32768; default thinking-mode sampling.
# [Timeouts & Locks] AISBench owns request timeouts; no locks during config load.
# [Side-Effect Policy] Defines configuration only; requests start through AISBench.
# Copy into AISBench's ais_bench/benchmark/configs/models/vllm_api/ directory.
# Run: ais_bench --models aisbench_qwen38 --datasets gpqa_gen_0_shot_str \
#      --mode all --dump-eval-details --merge-ds
# Defaults: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="qwen38-dp4-tp4-gpqa",
        path="/models/Qwen3.8-Flash-Next-w8a8-mtp",
        model="qwen3.8-flash-next",
        host_ip="127.0.0.1",
        host_port=8088,
        stream=False,
        request_rate=0,
        retry=2,
        max_out_len=32768,  # AISBench writes this as the API max_tokens field.
        batch_size=64,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            chat_template_kwargs=dict(enable_thinking=True),
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
