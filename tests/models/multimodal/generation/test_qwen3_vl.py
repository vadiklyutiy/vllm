# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....conftest import VIDEO_ASSETS
from .vlm_utils.builders import sample_frames_with_video_metadata

MODEL = "Qwen/Qwen3-VL-2B-Instruct"

VIDEO_PROMPTS = VIDEO_ASSETS.prompts(
    {
        "baby_reading": "<|im_start|>user\n<|vision_start|><|video_pad|>"
        "<|vision_end|>Describe this video in one sentence.<|im_end|>\n"
        "<|im_start|>assistant\n",
    }
)


@pytest.mark.core_model
def test_qwen3_vl_evs_functionality(vllm_runner, video_assets) -> None:
    """EVS must not break the M-RoPE recompute of the first prefill, where
    the prompt keeps the outer `<|vision_start|>` before the expanded video."""
    video = sample_frames_with_video_metadata(
        (video_assets[0].np_ndarrays, video_assets[0].metadata), 16
    )

    with vllm_runner(
        MODEL,
        runner="generate",
        max_model_len=4096,
        limit_mm_per_prompt={"video": 1},
        video_pruning_rate=0.75,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(VIDEO_PROMPTS, 32, videos=[video])

    output_ids, output_text = outputs[0]
    assert len(output_ids) > 0
    assert output_text
