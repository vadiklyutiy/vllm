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
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    recompute_mrope_positions,
)


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


@pytest.mark.parametrize("prompt_vision_markers", [False, True])
@pytest.mark.parametrize("chunk_starts_at_video", [False, True])
@pytest.mark.parametrize("interleave_text_tokens", [(0, 0), (1, 4)])
def test_match_qwen3vl_mrope_evs_on_prompt_vision_markers(
    prompt_vision_markers: bool,
    chunk_starts_at_video: bool,
    interleave_text_tokens: tuple[int, int],
):
    """Transformers>=5.10 Qwen3-VL processors expand only the video pad token
    and keep the prompt's own vision start / end tokens around the video, so
    the first frame's timestamp tokens follow a vision start token (#62937).
    Older processors replace those markers too."""
    hf_config = DummyConfig()
    grid_thw = (3, 4, 5)

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        *grid_thw,
        interleave_text_tokens=interleave_text_tokens,
        video_pruning_rate=0.5,
    )
    markers = prompt_vision_markers
    prefix_tokens = [1, 2, 3] + ([VISION_START_TOKEN_ID] if markers else [])
    suffix_tokens = ([VISION_END_TOKEN_ID] if markers else []) + [4, 5]
    input_tokens = prefix_tokens + video_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + video_tokens_pruned.tolist() + suffix_tokens
    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ]
    )

    def mrope_positions(tokens: list[int], offset: int) -> torch.Tensor:
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
            mm_position=PlaceholderRange(offset=offset, length=video_tokens.numel()),
        )
        positions, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=tokens, mm_features=[mm_feature], config=hf_config
        )
        return positions

    expected_mrope = mrope_positions(input_tokens, len(prefix_tokens))
    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # The embeddings carry [t, h, w, is_vision_start, is_video] per kept token.
    video_mrope = mrope_positions(video_tokens.tolist(), 0).permute(1, 0)
    mm_pos = torch.cat(
        [
            video_mrope[retention_mask],
            (video_tokens_pruned == VISION_START_TOKEN_ID)[:, None],
            (video_tokens_pruned == VIDEO_TOKEN_ID)[:, None],
        ],
        dim=1,
    )
    video_embeddings = torch.cat(
        [torch.empty((len(video_tokens_pruned), 16)), mm_pos.float()], dim=1
    )

    computed_mrope = torch.empty((3, len(input_tokens_pruned)), dtype=torch.long)
    computed_mrope[:, : len(prefix_tokens)] = expected_mrope[:, : len(prefix_tokens)]

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=[video_embeddings],
        mrope_positions=computed_mrope,
        num_computed_tokens=len(prefix_tokens) if chunk_starts_at_video else 0,
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_end_token_id=hf_config.vision_end_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope_masked)


@pytest.mark.parametrize(
    ("num_channels", "media_token_id"), [(4, VIDEO_TOKEN_ID), (5, IMAGE_TOKEN_ID)]
)
def test_recompute_mrope_chunk_starts_at_media_without_timestamps(
    num_channels: int, media_token_id: int
):
    """A chunk starts at the placeholder when the encoder budget cannot fit the
    item in the step that computed its vision start token (Qwen2.5-VL video:
    4 channels, Qwen3-VL image: 5 channels without timestamps)."""
    mm_pos = compute_mrope_for_media(torch.tensor([3, 2, 2]), spatial_merge_size=1)
    mm_pos = mm_pos[torch.tensor([True] * 4 + [True, False] * 4)].permute(1, 0)
    if num_channels == 5:
        mm_pos = torch.cat([mm_pos, torch.zeros_like(mm_pos[:1])])
    prefix_tokens = [1, 2, VISION_START_TOKEN_ID]
    input_ids = torch.tensor(
        prefix_tokens + [media_token_id] * mm_pos.shape[1] + [VISION_END_TOKEN_ID, 4, 5]
    )
    mrope_positions = torch.arange(len(input_ids)).expand(3, -1).clone()

    def recompute(num_computed_tokens: int) -> tuple[torch.Tensor, int]:
        return recompute_mrope_positions(
            input_ids,
            [mm_pos],
            mrope_positions,
            num_computed_tokens,
            VISION_START_TOKEN_ID,
            IMAGE_TOKEN_ID,
            VIDEO_TOKEN_ID,
        )

    expected, expected_delta = recompute(0)
    actual, actual_delta = recompute(len(prefix_tokens))

    assert torch.equal(actual, expected)
    assert actual_delta == expected_delta


@pytest.mark.parametrize("num_channels", [4, 5])
def test_recompute_mrope_two_images_in_one_chunk(num_channels: int):
    """The first image has as many embeddings as there are tokens up to and
    including its vision start token. The second image of the same chunk must
    still go after its own vision start token."""
    sides = (4, 3)
    input_ids = [1] * (sides[0] ** 2 - 1)
    mm_features, mm_pos = [], []
    for side in sides:
        input_ids += [VISION_START_TOKEN_ID]
        grid_thw = torch.tensor([1, side, side])
        mm_features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem(
                    {"image_grid_thw": MultiModalFieldElem(data=grid_thw, field=None)}
                ),
                modality="image",
                identifier="DUMMY",
                mm_position=PlaceholderRange(offset=len(input_ids), length=side**2),
            )
        )
        input_ids += [IMAGE_TOKEN_ID] * side**2 + [VISION_END_TOKEN_ID, 2]
        pos = compute_mrope_for_media(grid_thw, spatial_merge_size=1).permute(1, 0)
        if num_channels == 5:
            pos = torch.cat([pos, torch.zeros_like(pos[:1])])
        mm_pos.append(pos)

    expected, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_ids, mm_features=mm_features, config=DummyConfig()
    )
    actual, _ = recompute_mrope_positions(
        torch.tensor(input_ids),
        mm_pos,
        torch.arange(len(input_ids)).expand(3, -1).clone(),
        0,
        VISION_START_TOKEN_ID,
        IMAGE_TOKEN_ID,
        VIDEO_TOKEN_ID,
    )

    assert torch.equal(actual, expected)
