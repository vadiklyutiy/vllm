# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams

MODEL_PATH = "ai21labs/Jamba-tiny-dev"


def test_jamba_lora_mixed_batch():
    """With max_num_seqs below the largest CUDA graph capture size, startup
    capture runs mixed prefill/decode batches with LoRAs active on the Mamba
    mixer projections."""
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_lora_rank=8,
        max_num_seqs=256,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
    )
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=8),
        use_tqdm=False,
    )
    assert len(outputs[0].outputs[0].token_ids) > 0
