# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pixtral's multimodal processor with the Mistral tokenizer."""

from io import BytesIO

import pybase64
import pytest
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import ExifTags, Image

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.processing import BaseMultiModalProcessor

from ...utils import build_model_context

_MODEL_ID = "mistralai/Pixtral-12B-2409"


def _jpeg_with_orientation(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    with BytesIO() as buffer:
        Image.new("RGB", (640, 480)).save(buffer, "JPEG", exif=exif)
        return buffer.getvalue()


def _assert_image_tokens_match_processed_images(
    processor: BaseMultiModalProcessor,
    prompt: list[int],
    images: list[object],
) -> None:
    mm_items = processor.info.parse_mm_data({"image": images})
    processed = processor(prompt, mm_items=mm_items)

    tokenizer = processor.info.get_tokenizer()
    processed_images = [mm_items["image"].get(i) for i in range(len(images))]
    request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[ImageChunk(image=image) for image in processed_images])
        ]
    )
    expected_prompt = tokenizer.mistral.encode_chat_completion(request).tokens
    assert processed["prompt_token_ids"] == expected_prompt

    placeholders = processed["mm_placeholders"]["image"]
    assert len(placeholders) == len(images)
    for placeholder, image in zip(placeholders, processed_images):
        expected_tokens = tokenizer.instruct.mm_encoder(ImageChunk(image=image)).tokens
        start_idx = placeholder.offset
        end_idx = start_idx + placeholder.length
        assert processed["prompt_token_ids"][start_idx:end_idx] == expected_tokens


@pytest.mark.parametrize("orientation", [1, 6])
def test_chat_image_tokens_follow_exif_orientation(orientation: int):
    """mistral-common sizes the chat template's image tokens without applying
    the EXIF orientation, while vLLM processes the EXIF-transposed image."""
    ctx = build_model_context(_MODEL_ID)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    image_bytes = _jpeg_with_orientation(orientation)
    image_url = "data:image/jpeg;base64," + pybase64.b64encode(image_bytes).decode()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            }
        ]
    )

    _assert_image_tokens_match_processed_images(
        processor, prompt, [ImageMediaIO().load_bytes(image_bytes)]
    )


@pytest.mark.parametrize("orientation", [1, 6])
def test_token_prompt_image_tokens_follow_exif_orientation(orientation: int):
    """Same as above for a token prompt encoded by mistral-common from PIL
    images, which vLLM EXIF-transposes when parsing the multimodal data. The
    two images are next to each other, so their runs are replaced in order."""
    ctx = build_model_context(_MODEL_ID)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    images = [
        Image.open(BytesIO(_jpeg_with_orientation(orientation))),
        Image.open(BytesIO(_jpeg_with_orientation(1))),
    ]
    request = ChatCompletionRequest(
        messages=[UserMessage(content=[ImageChunk(image=image) for image in images])]
    )
    prompt = tokenizer.mistral.encode_chat_completion(request).tokens

    _assert_image_tokens_match_processed_images(processor, prompt, images)
