# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.clip import CLIPVisionEmbeddings
from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


class _StubModel:
    """Carries only the attributes `get_mm_lora_token_counts` reads from
    `self`, so it can run without building the full `nn.Module`."""

    def __init__(self, config):
        self.config = config


@pytest.mark.parametrize("model_id", ["llava-hf/llava-1.5-7b-hf"])
@pytest.mark.parametrize("num_images", [1, 10])
def test_mm_lora_tower_tokens_include_class_token(
    default_vllm_config, model_id, num_images
):
    """The CLIP tower runs each image with its class token, one row more than
    the image's placeholders, so the tower LoRA token count must include it.
    10 images fill a `--max-num-batched-tokens 5760` encoder budget."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    config = ctx.model_config.hf_config
    info = MULTIMODAL_REGISTRY.create_processor(ctx.model_config).info
    image_size = info.get_image_size_with_most_features()
    num_embeds_per_image = info.get_max_image_tokens()

    pixel_values = torch.zeros(1, 3, image_size.height, image_size.width)
    embeddings = CLIPVisionEmbeddings(config.vision_config)(pixel_values)
    num_tower_rows_per_image = embeddings.shape[1]

    tower_tokens, connector_tokens = (
        LlavaForConditionalGeneration.get_mm_lora_token_counts(
            _StubModel(config),
            modality="image",
            mm_kwargs=None,
            num_mm_embeds=num_images * num_embeds_per_image,
        )
    )

    assert tower_tokens == num_images * num_tower_rows_per_image
    assert connector_tokens == num_images * num_embeds_per_image
