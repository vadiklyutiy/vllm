# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.utils.mistral import is_mistral_tokenizer


def test_mistral_thinking_budget_forces_end_think_control_token(vllm_runner):
    """Under `tokenizer_mode="mistral"` the budget must force the `[/THINK]`
    control token, so it must find `[THINK]` in the prompt as one."""
    with vllm_runner(
        "mistralai/Magistral-Small-2509",
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="dummy",
        hf_overrides={"text_config": {"num_hidden_layers": 1}},
        reasoning_parser="mistral",
        limit_mm_per_prompt={"image": 0},
        max_model_len=256,
        kv_cache_memory_bytes=16 * 1024**2,
        enforce_eager=True,
    ) as runner:
        tokenizer = runner.llm.get_tokenizer()
        assert is_mistral_tokenizer(tokenizer)
        # The dummy model repeats the last prompt token. End on text, since a
        # repeated `[THINK]` would restart the budget count.
        prompt = [
            tokenizer.bos_token_id,
            tokenizer.instruct.BEGIN_INST,
            *tokenizer.encode("What is 2+2?", add_special_tokens=False),
            tokenizer.instruct.END_INST,
            tokenizer.instruct.BEGIN_THINK,
            *tokenizer.encode("Okay", add_special_tokens=False),
        ]
        budget = 4
        output = runner.llm.generate(
            {"prompt_token_ids": prompt},
            SamplingParams(
                max_tokens=budget + 2,
                temperature=0.0,
                ignore_eos=True,
                thinking_token_budget=budget,
            ),
        )[0].outputs[0]

    assert tokenizer.instruct.END_THINK in output.token_ids[:budget]
