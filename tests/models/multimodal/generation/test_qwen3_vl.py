# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from ...utils import check_logprobs_close

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video in one sentence.<|im_end|>\n<|im_start|>assistant\n"
)


@pytest.mark.core_model
def test_qwen3_vl_video_pruning_matches_unpruned(vllm_runner) -> None:
    """Pruning keeps every token of a one-frame video but still recomputes its
    M-RoPE positions, which must match the unpruned ones. Transformers>=5.10
    processors keep the prompt's own vision start token in front of the video,
    which the recompute took for the first frame's own and then crashed
    (#62937)."""
    num_frames = 2  # one temporal patch
    video = np.random.default_rng(0).integers(
        0, 256, size=(num_frames, 224, 224, 3), dtype=np.uint8
    )
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }

    outputs = {}
    for video_pruning_rate in (None, 0.75):
        with vllm_runner(
            MODEL,
            runner="generate",
            max_model_len=4096,
            limit_mm_per_prompt={"image": 0, "video": 1},
            video_pruning_rate=video_pruning_rate,
        ) as vllm_model:
            outputs[video_pruning_rate] = vllm_model.generate_greedy_logprobs(
                [VIDEO_PROMPT],
                max_tokens=16,
                num_logprobs=5,
                videos=[(video, metadata)],
            )

    check_logprobs_close(
        outputs_0_lst=outputs[None],
        outputs_1_lst=outputs[0.75],
        name_0="unpruned",
        name_1="pruned",
    )
