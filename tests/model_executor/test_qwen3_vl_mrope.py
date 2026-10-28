# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune.evs import compute_mrope_for_media


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    # _get_mrope_input_positions returns CPU tensors (via torch.from_numpy).
    # Ensure the default device is CPU so the rest of the test tensors match.
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass
class DummyConfig:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: DummyVisionConfig = dataclasses.field(
        default_factory=DummyVisionConfig
    )


def make_video_embedding(
    t, h, w, interleave_text_tokens: tuple[int, int], video_pruning_rate: float = 0.0
):
    """Make a video embedding for a given video size and pruning rate.

    Args:
        t: Number of frames.
        h: Number of rows.
        w: Number of columns.
        interleave_text_tokens: Tuple of minimum and maximum number of text tokens to
            interleave with the video.
        video_pruning_rate: Pruning rate for the video.

    Returns:
        Tuple of (unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask)

    """
    unpruned_tokens_sequence = []
    population = list(range(1, 100))

    for _ in range(t):
        num_prefix_tokens = random.randint(
            interleave_text_tokens[0], interleave_text_tokens[1]
        )

        prefix_tokens = random.choices(population, k=num_prefix_tokens)
        vision_tokens = (
            [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * h * w + [VISION_END_TOKEN_ID]
        )

        unpruned_tokens_sequence.extend(prefix_tokens)
        unpruned_tokens_sequence.extend(vision_tokens)

    unpruned_tokens_sequence = torch.tensor(unpruned_tokens_sequence, dtype=torch.long)
    video_token_mask = unpruned_tokens_sequence == VIDEO_TOKEN_ID

    pruning_mask = torch.bernoulli(video_token_mask.float() * video_pruning_rate).bool()
    # Sanity check that we don't prune what should not be pruned.
    assert not pruning_mask[~video_token_mask].any()

    retention_mask = ~pruning_mask
    pruned_tokens_sequence = unpruned_tokens_sequence[retention_mask]
    return unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_thw", [[3, 8, 7], [128, 10, 12]])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
@pytest.mark.parametrize("num_suffix_tokens", [0, 7])
@pytest.mark.parametrize("video_pruning_rate", [0, 0.25, 0.75])
@pytest.mark.parametrize("interleave_text_tokens", [(0, 0), (1, 4)])
def test_match_qwen3vl_mrope_evs_on(
    spatial_merge_size: int,
    num_prefix_tokens: int,
    grid_thw: tuple[int, int, int],
    num_suffix_tokens: int,
    video_pruning_rate: float,
    interleave_text_tokens: tuple[int, int],
):
    hf_config = DummyConfig()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size

    t, h, w = grid_thw
    population = list(range(1, 100))
    prefix_tokens = random.choices(population, k=num_prefix_tokens)
    suffix_tokens = random.choices(population, k=num_suffix_tokens)

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t,
        h // spatial_merge_size,
        w // spatial_merge_size,
        interleave_text_tokens=interleave_text_tokens,
        video_pruning_rate=video_pruning_rate,
    )
    assert len(video_tokens) == len(retention_mask)

    input_tokens = prefix_tokens + video_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + video_tokens_pruned.tolist() + suffix_tokens

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ],
        dim=0,
    )

    # Build the GT mrope for unpruned input.
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=len(input_tokens)),
    )
    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[mm_feature],
        config=hf_config,
    )

    # Compute mrope for a video-only media (unpruned).
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=video_tokens.numel()),
    )
    video_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=video_tokens.tolist(),
        mm_features=[mm_feature],
        config=hf_config,
    )
    video_mrope = video_mrope.permute(1, 0)  # [N, 3]
    hidden_size = 16

    is_video_embed = torch.isin(
        video_tokens_pruned, torch.tensor([VIDEO_TOKEN_ID], dtype=torch.long)
    )

    expanded_positions = torch.full(
        (len(video_tokens_pruned), 5),
        fill_value=-100,
        device=video_mrope.device,
        dtype=torch.long,
    )
    expanded_positions[is_video_embed, :3] = video_mrope[retention_mask][is_video_embed]
    expanded_positions[~is_video_embed, :3] = video_mrope[retention_mask][
        ~is_video_embed
    ]

    is_vision_start = video_tokens_pruned == VISION_START_TOKEN_ID
    expanded_positions[..., 3] = is_vision_start
    expanded_positions[..., 4] = is_video_embed

    # Check that all positions were filled, since we initialized them as negative.
    assert (expanded_positions >= 0).all()

    video_embeddings = torch.empty(
        (len(video_tokens_pruned), hidden_size), device=video_mrope.device
    )

    video_embeddings = torch.cat(
        [
            video_embeddings,
            expanded_positions.float(),
        ],
        dim=1,
    )
    multimodal_embeddings = [video_embeddings]

    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # Initialize computed_mrope with sequential positions for all prefix tokens
    computed_mrope = torch.empty((3, len(input_tokens_pruned)), dtype=torch.long)
    computed_mrope[:, 0 : len(prefix_tokens)] = expected_mrope[
        :, 0 : len(prefix_tokens)
    ]

    # Paranoia check that computed_mrope is wrong.
    assert not torch.equal(computed_mrope, expected_mrope_masked)

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=multimodal_embeddings,
        mrope_positions=computed_mrope,
        num_computed_tokens=len(prefix_tokens),
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope_masked)


def _mm_feature(
    modality: str, grid_thw: list[int], offset: int, length: int
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                f"{modality}_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality=modality,
        identifier=modality,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


@pytest.mark.parametrize("start_at_placeholder", [False, True])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
def test_match_qwen3vl_mrope_evs_on_outer_vision_markers(
    num_prefix_tokens: int, start_at_placeholder: bool
):
    """Transformers>=5.10 Qwen3-VL processors keep the prompt's outer
    `<|vision_start|>`/`<|vision_end|>` around the expanded video, so the first
    vision start token of the prompt does not belong to the first frame.
    A prefix cache hit or a chunk can end right before the placeholder."""
    random.seed(0)
    torch.manual_seed(0)
    hf_config = DummyConfig()
    grid_thw = [3, 8, 7]
    t, h, w = grid_thw

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t, h, w, interleave_text_tokens=(1, 4), video_pruning_rate=0.75
    )
    head = random.choices(range(1, 100), k=num_prefix_tokens) + [VISION_START_TOKEN_ID]
    tail = [VISION_END_TOKEN_ID] + random.choices(range(1, 100), k=5)
    input_tokens = head + video_tokens.tolist() + tail
    input_tokens_pruned = head + video_tokens_pruned.tolist() + tail

    def mrope(tokens: list[int], offset: int) -> torch.Tensor:
        mm_feature = _mm_feature("video", grid_thw, offset, len(video_tokens))
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=tokens, mm_features=[mm_feature], config=hf_config
        )[0]

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(head), dtype=torch.bool),
            retention_mask,
            torch.ones(len(tail), dtype=torch.bool),
        ]
    )
    expected_mrope = mrope(input_tokens, len(head))[:, whole_sequence_retention_mask]

    expanded_positions = torch.cat(
        [
            mrope(video_tokens.tolist(), 0)[:, retention_mask],
            (video_tokens_pruned == VISION_START_TOKEN_ID)[None],
            (video_tokens_pruned == VIDEO_TOKEN_ID)[None],
        ]
    ).T
    video_embeddings = torch.cat(
        [torch.empty((len(video_tokens_pruned), 16)), expanded_positions.float()],
        dim=1,
    )

    # Only the text before the video has its final positions.
    computed_mrope = torch.full_like(expected_mrope, -1)
    computed_mrope[:, : len(head)] = expected_mrope[:, : len(head)]

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=[video_embeddings],
        mrope_positions=computed_mrope,
        num_computed_tokens=len(head) if start_at_placeholder else 0,
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope)


@pytest.mark.parametrize("outer_vision_markers", [False, True])
def test_match_qwen3vl_mrope_evs_on_image_before_video(outer_vision_markers: bool):
    """An image and a pruned video recomputed in the same prefill: the video
    must not be located inside the image placed before it."""
    random.seed(0)
    torch.manual_seed(0)
    hf_config = DummyConfig()
    image_grid_thw, video_grid_thw = [1, 4, 4], [3, 8, 7]
    num_image_tokens = 16
    t, h, w = video_grid_thw

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t, h, w, interleave_text_tokens=(1, 4), video_pruning_rate=0.75
    )
    head = (
        [1, VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_image_tokens
        + [VISION_END_TOKEN_ID]
        + [VISION_START_TOKEN_ID] * outer_vision_markers
    )
    tail = [VISION_END_TOKEN_ID] * outer_vision_markers + [2, 3, 4]
    input_tokens = head + video_tokens.tolist() + tail
    input_tokens_pruned = head + video_tokens_pruned.tolist() + tail

    expected_mrope = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[
            _mm_feature("image", image_grid_thw, 2, num_image_tokens),
            _mm_feature("video", video_grid_thw, len(head), len(video_tokens)),
        ],
        config=hf_config,
    )[0][
        :,
        torch.cat(
            [
                torch.ones(len(head), dtype=torch.bool),
                retention_mask,
                torch.ones(len(tail), dtype=torch.bool),
            ]
        ),
    ]

    image_positions = compute_mrope_for_media(image_grid_thw, 1)
    image_embeddings = torch.cat(
        [
            torch.empty((num_image_tokens, 16)),
            image_positions.float(),
            torch.zeros((num_image_tokens, 1)),
        ],
        dim=1,
    )
    video_mrope = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=video_tokens.tolist(),
        mm_features=[_mm_feature("video", video_grid_thw, 0, len(video_tokens))],
        config=hf_config,
    )[0]
    video_positions = torch.cat(
        [
            video_mrope[:, retention_mask],
            (video_tokens_pruned == VISION_START_TOKEN_ID)[None],
            (video_tokens_pruned == VIDEO_TOKEN_ID)[None],
        ]
    ).T
    video_embeddings = torch.cat(
        [torch.empty((len(video_tokens_pruned), 16)), video_positions.float()],
        dim=1,
    )

    computed_mrope = torch.full_like(expected_mrope, -1)
    computed_mrope[:, :2] = expected_mrope[:, :2]

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=[image_embeddings, video_embeddings],
        mrope_positions=computed_mrope,
        num_computed_tokens=0,
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope)


@pytest.mark.parametrize("lumped_prompt", [False, True])
@pytest.mark.parametrize("outer_vision_markers", [False, True])
def test_match_qwen3vl_mrope_evs_on_empty_first_frame(
    outer_vision_markers: bool, lumped_prompt: bool
):
    """Pruning can remove every video token of the first frame. vLLM's prompt
    keeps all retained video tokens in the first frame; a prompt may also keep
    the per-frame layout of the embedding."""
    hf_config = DummyConfig()
    grid_thw = [2, 2, 2]
    ts0, ts1 = [11, 12], [13, 14]
    frame = [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID]
    video_tokens = ts0 + frame + ts1 + frame
    retention_mask = torch.tensor([True] * 3 + [False] * 4 + [True] * 9)
    embedding_tokens = torch.tensor(video_tokens)[retention_mask]
    prompt_tokens = (
        ts0 + frame + ts1 + [VISION_START_TOKEN_ID, VISION_END_TOKEN_ID]
        if lumped_prompt
        else embedding_tokens.tolist()
    )
    head = [1, 2, 3] + [VISION_START_TOKEN_ID] * outer_vision_markers
    tail = [VISION_END_TOKEN_ID] * outer_vision_markers + [4, 5]

    def mrope(tokens: list[int], offset: int) -> torch.Tensor:
        mm_feature = _mm_feature("video", grid_thw, offset, len(video_tokens))
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=tokens, mm_features=[mm_feature], config=hf_config
        )[0]

    expected_mrope = mrope(head + video_tokens + tail, len(head))[
        :,
        torch.cat(
            [
                torch.ones(len(head), dtype=torch.bool),
                retention_mask,
                torch.ones(len(tail), dtype=torch.bool),
            ]
        ),
    ]
    video_positions = torch.cat(
        [
            mrope(video_tokens, 0)[:, retention_mask],
            (embedding_tokens == VISION_START_TOKEN_ID)[None],
            (embedding_tokens == VIDEO_TOKEN_ID)[None],
        ]
    ).T
    video_embeddings = torch.cat(
        [torch.empty((len(embedding_tokens), 16)), video_positions.float()], dim=1
    )

    computed_mrope = torch.full_like(expected_mrope, -1)
    computed_mrope[:, : len(head)] = expected_mrope[:, : len(head)]

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=head + prompt_tokens + tail,
        multimodal_embeddings=[video_embeddings],
        mrope_positions=computed_mrope,
        num_computed_tokens=0,
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope)
