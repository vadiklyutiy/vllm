# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest
import torch

from tests.models.utils import check_logprobs_close, dummy_hf_overrides
from vllm.assets.image import ImageAsset
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.models.radio import RadioModel, calc_seq_lens
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.utils.torch_utils import set_default_torch_dtype


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
@torch.inference_mode()
def test_radio_dynamic_resolution_compilation(dist_init):
    """The compiled RADIO encoder layer is traced once, on a packed batch of
    two images. Packed batches with other image counts or sizes, and conv3d
    video clips with other tubelet counts, must reuse that graph and match
    the eager encoder."""
    radio_config = RadioConfig(
        model_name="vit_small_patch16_224",
        register_multiple=8,
        video_temporal_patch_size=2,
    )
    patch_size = radio_config.patch_size

    def build_model(compile_mm_encoder: bool) -> tuple[VllmConfig, RadioModel]:
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                compile_mm_encoder=compile_mm_encoder,
                cudagraph_mode=CUDAGraphMode.NONE,
            )
        )
        with (
            set_current_vllm_config(vllm_config),
            set_default_torch_dtype(torch.bfloat16),
        ):
            model = RadioModel(radio_config, num_hidden_layers_override=2)
        return vllm_config, model.to("cuda")

    torch.manual_seed(0)
    compiled_config, compiled_model = build_model(compile_mm_encoder=True)
    eager_config, eager_model = build_model(compile_mm_encoder=False)
    for param in compiled_model.parameters():
        param.normal_(std=0.02)
    # Unit scales, so that a wrong attention mask shows in the output.
    for layer in compiled_model.model.encoder.layers:
        for param in (layer.ls1, layer.ls2, layer.norm1.weight, layer.norm2.weight):
            param.fill_(1.0)
    eager_model.load_state_dict(compiled_model.state_dict())

    for imgs_sizes in (
        [(320, 256), (224, 448)],
        [(256, 384)],
        [(160, 160), (192, 320), (128, 256)],
        [(448, 160), (128, 384)],
    ):
        num_patches = sum(calc_seq_lens(imgs_sizes, patch_size))
        pixel_values = torch.randn(
            1, num_patches, 3 * patch_size**2, dtype=torch.bfloat16, device="cuda"
        )
        with set_current_vllm_config(eager_config):
            expected = eager_model(pixel_values, imgs_sizes=imgs_sizes)
        with set_current_vllm_config(compiled_config):
            actual = compiled_model(pixel_values, imgs_sizes=imgs_sizes)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    for num_frames in (6, 2):
        pixel_values = torch.randn(
            num_frames, 3, 224, 224, dtype=torch.bfloat16, device="cuda"
        )
        with set_current_vllm_config(eager_config):
            expected = eager_model(pixel_values, num_frames=num_frames)
        with set_current_vllm_config(compiled_config):
            actual = compiled_model(pixel_values, num_frames=num_frames)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073
@pytest.mark.forked
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_nemotron_parse_35_compilation(vllm_runner):
    """Parse 3.5 profiles its encoder with several images, so the compiled
    RADIO layers are traced on a packed batch. A one-image request must match
    the eager encoder."""
    prompts = [
        "<pad></s><s><predict_bbox><predict_classes><output_markdown>"
        "<predict_no_text_in_pic>"
    ]
    images = [ImageAsset("paper-11").pil_image_ext(ext="png").convert("RGB")]
    outputs = []
    for compile_mm_encoder in (False, True):
        with vllm_runner(
            "nvidia/Nemotron-3.5-Parse",
            trust_remote_code=True,
            max_model_len=16384,
            limit_mm_per_prompt={"image": 1},
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": compile_mm_encoder,
            },
        ) as vllm_model:
            outputs.append(
                vllm_model.generate_greedy_logprobs(
                    prompts, max_tokens=32, num_logprobs=5, images=images
                )
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="eager_encoder",
        name_1="compiled_encoder",
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
