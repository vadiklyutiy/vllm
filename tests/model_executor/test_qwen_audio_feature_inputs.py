# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration


@pytest.mark.parametrize(
    "model_cls",
    [Qwen3ASRForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration],
)
def test_parse_padded_audio_features_with_different_lengths(model_cls) -> None:
    """Rust frontend items padded to different lengths are batched as a list."""
    n_mels = 128
    padded_lengths = [6, 9]
    lengths = [4, 9]
    features = [torch.randn(n_mels, n) for n in padded_lengths]
    masks = [torch.ones(n, dtype=torch.long) for n in padded_lengths]

    audio_input = model_cls._parse_and_validate_audio_input(
        None,
        input_audio_features=features,
        audio_feature_lengths=torch.tensor(lengths),
        feature_attention_mask=masks,
    )

    expected = torch.cat(
        [feature[:, :length] for feature, length in zip(features, lengths)], dim=-1
    )
    torch.testing.assert_close(audio_input["input_features"], expected)
