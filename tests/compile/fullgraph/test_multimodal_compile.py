# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest
import torch
from transformers import AutoProcessor

from tests.models.utils import (
    check_embeddings_close,
    check_logprobs_close,
    dummy_hf_overrides,
)
from vllm import SamplingParams
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform


def test_compile():
    vllm_config = VllmConfig()
    # Default configuration does not compile mm encoder
    assert not vllm_config.compilation_config.compile_mm_encoder


@pytest.mark.forked
@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
@pytest.mark.parametrize(
    ("model", "tokenizer", "config_format", "tokenizer_mode", "model_arch"),
    [
        pytest.param(
            "mistralai/Pixtral-12B-2409",
            None,
            "mistral",
            "mistral",
            "PixtralForConditionalGeneration",
            id="pixtral",
        ),
        pytest.param(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "mistral-community/pixtral-12b",
            "hf",
            "auto",
            "Mistral3ForConditionalGeneration",
            id="mistral3-hf",
        ),
    ],
)
@pytest.mark.parametrize(
    ("compile_mm_encoder", "expected_models"),
    [
        # The dummy model has two compiled text components. Enabling encoder
        # compilation adds its single vision transformer block.
        pytest.param(False, 2, id="disabled"),
        pytest.param(True, 3, id="enabled"),
    ],
)
def test_pixtral_compilation(
    vllm_runner,
    monkeypatch,
    model: str,
    tokenizer: str | None,
    config_format: str,
    tokenizer_mode: str,
    model_arch: str,
    compile_mm_encoder: bool,
    expected_models: int,
):
    """Test compilation for both Pixtral vision encoder implementations."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    with (
        compilation_counter.expect(num_models_seen=expected_models),
        vllm_runner(
            model,
            tokenizer_name=tokenizer,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            load_format="dummy",
            hf_overrides=partial(dummy_hf_overrides, model_arch=model_arch),
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.8,
            attention_backend="FLASH_ATTN",
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": compile_mm_encoder,
                "cudagraph_mode": CUDAGraphMode.NONE,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_qwen2_5_vl_compilation(vllm_runner, monkeypatch):
    """Test that Qwen2.5-VL vision submodules are compiled.

    This test verifies that the 3 vision submodules (Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionBlock, and Qwen2_5_VisionPatchMerger) are properly tagged
    for compilation by checking that num_models_seen increases by at least 3.
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        # NOTE: Qwen2.5-VL has 35 models in total - the LLM backend
        # Vision Patch Embed, Vision Patch Merger, and then 32 Vision Blocks
        # (one for each layer) - in the future, we should fix vLLM compilation
        # logic to handle this case and only compile the Vision submodules once
        # and reuse the compiled code for all layers
        # See https://github.com/vllm-project/vllm/issues/27590
        compilation_counter.expect(num_models_seen=35),
        vllm_runner(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": True,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_qwen2_5_vl_no_vit_compilation(vllm_runner, monkeypatch):
    """Test that Qwen2.5-VL vision submodules are not compiled when the
    config is passed off
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(num_models_seen=1),
        vllm_runner(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": False,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
@pytest.mark.parametrize(
    ("compile_mm_encoder", "expected_models"),
    [
        # The Ernie-4.5 language model always registers two compiled modules.
        # Enabling encoder compilation adds one per SigLIP encoder layer (27).
        pytest.param(False, 2, id="disabled"),
        pytest.param(True, 29, id="enabled"),
    ],
)
def test_paddleocr_vl_compilation(
    vllm_runner, monkeypatch, compile_mm_encoder: bool, expected_models: int
):
    """Test that PaddleOCR-VL encoder layers compile only when enabled."""
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        compilation_counter.expect(num_models_seen=expected_models),
        vllm_runner(
            "PaddlePaddle/PaddleOCR-VL",
            trust_remote_code=True,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.5,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": compile_mm_encoder,
            },
        ) as _,
    ):
        pass


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
@pytest.mark.parametrize(
    ("model", "max_num_batched_tokens"),
    [
        # Encoder `pixel_values` is unannotated, `patch_attention_mask` a BoolTensor
        pytest.param("HuggingFaceTB/SmolVLM-256M-Instruct", None, id="smolvlm"),
        # Encoder packs all images into one sequence and derives where each starts
        # from `grid_thw`, while traced with one or with two dummy images
        pytest.param("Qwen/Qwen3-VL-2B-Instruct", None, id="qwen3_vl"),
        pytest.param("Qwen/Qwen3-VL-2B-Instruct", 32768, id="qwen3_vl-2-dummies"),
    ],
)
def test_transformers_backend_encoder_compilation(
    vllm_runner,
    image_assets,
    monkeypatch,
    model: str,
    max_num_batched_tokens: int | None,
):
    """Test that the Transformers modeling backend compiles the encoder, and that
    the compiled encoder matches the eager one on a batch of two images."""
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Different sizes, so the items packed into one sequence differ in length
    sizes = [(448, 448), (336, 448)]
    images = [asset.pil_image.resize(size) for asset, size in zip(image_assets, sizes)]
    content = [{"type": "image_pil", "image_pil": image} for image in images]
    content.append({"type": "text", "text": "Describe the two images."})
    messages = [{"role": "user", "content": content}]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    image_inputs = AutoProcessor.from_pretrained(model).image_processor(
        images, return_tensors="pt"
    )

    def embed_images(vllm_model) -> list[list[float]]:
        hf_model = vllm_model.model
        inputs = {k: v.to(hf_model.device) for k, v in image_inputs.items()}
        with torch.inference_mode():
            output = hf_model.get_image_features(**inputs, return_dict=True)
        return [embeds.flatten().tolist() for embeds in output.pooler_output]

    def chat(compile_mm_encoder: bool):
        with vllm_runner(
            model,
            model_impl="transformers",
            max_model_len=4096,
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"image": 2},
            gpu_memory_utilization=0.5,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": compile_mm_encoder,
            },
        ) as vllm_model:
            (output,) = vllm_model.llm.chat(messages, sampling_params)
            (image_embeds,) = vllm_model.llm.apply_model(embed_images)
        completion = output.outputs[0]
        outputs = [(list(completion.token_ids), completion.text, completion.logprobs)]
        return outputs, image_embeds

    # Compiled first: decorating the classes is global, so later runs count more
    with compilation_counter.expect(num_models_seen=2):
        compiled, compiled_embeds = chat(compile_mm_encoder=True)
    eager, eager_embeds = chat(compile_mm_encoder=False)

    check_logprobs_close(
        outputs_0_lst=eager,
        outputs_1_lst=compiled,
        name_0="eager_encoder",
        name_1="compiled_encoder",
    )
    check_embeddings_close(
        embeddings_0_lst=eager_embeds,
        embeddings_1_lst=compiled_embeds,
        name_0="eager_encoder",
        name_1="compiled_encoder",
        # Inductor's bf16 kernels round differently from eager
        tol=1e-2,
    )


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
# Requires Cuda and 8 gpus as well
@pytest.mark.forked
@pytest.mark.skip(reason="Skipping due to CI resource constraints")
def test_mllama4_vit_compilation(vllm_runner, monkeypatch):
    """Test that Mllama4 vision submodules are compiled.

    This test verifies that the 2 vision submodules (Llama4VisionEncoder,
    Llama4VisionPixelShuffleMLP) are properly tagged
    for compilation by checking that num_models_seen increases to 3.

    However since we are using TP=8, we compilation_counter will not
    work properly so we will just check the run succeeds rn
    """
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with (
        monkeypatch.context(),
        # TODO: Since we require TP=8, this messes with the compilation
        # counter. We should fix this in the future, but leave for now
        # to make sure that compilation runs (no crash) with llama vision encoder
        compilation_counter.expect(num_models_seen=0),
        vllm_runner(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            max_model_len=512,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=8,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": True,
            },
        ),
    ):
        pass
