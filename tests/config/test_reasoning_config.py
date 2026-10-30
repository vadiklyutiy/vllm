# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.tokenizers import cached_tokenizer_from_config


@pytest.fixture(scope="module")
def mistral_model_config() -> ModelConfig:
    return ModelConfig(model="mistralai/Magistral-Small-2509", tokenizer_mode="mistral")


@pytest.mark.parametrize(
    "transition", ["", "I have to answer now."], ids=["parser_default", "transition"]
)
def test_mistral_reasoning_token_ids_are_control_tokens(
    mistral_model_config: ModelConfig, transition: str
):
    """The Mistral tokenizer encodes `[THINK]` text as plain text, so the
    thinking budget must still get the control tokens the model emits."""
    config = ReasoningConfig(
        reasoning_parser="mistral",
        reasoning_end_str=f"{transition}[/THINK]" if transition else "",
    )
    config.initialize_token_ids(mistral_model_config)

    tokenizer = cached_tokenizer_from_config(mistral_model_config)
    end_think = tokenizer.instruct.END_THINK
    assert config.reasoning_end_token_ids == [
        *tokenizer.encode(transition, add_special_tokens=False),
        end_think,
    ]
    assert config.natural_reasoning_end_token_ids == [end_think]
    assert config.reasoning_start_token_ids == [tokenizer.instruct.BEGIN_THINK]
