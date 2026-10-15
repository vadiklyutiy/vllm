# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.warmup.mamba_triton_warmup import _warm_batch_memcpy_kernel
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test

# Kernels startup warmup must compile for every batch shape.
_WARMED_KERNELS = frozenset(
    {
        "precopy_mamba_align_fused_kernel",
        "postprocess_mamba_fused_kernel",
        "_causal_conv1d_fwd_kernel",
    }
)
_NGRAM_SPEC_CONFIG = {
    "method": "ngram",
    "num_speculative_tokens": 3,
    "prompt_lookup_max": 3,
}
_LFM2 = (
    "LiquidAI/LFM2-8B-A1B",
    {"num_hidden_layers": 3, "layer_types": ["conv", "conv", "full_attention"]},
)
_GRANITE_4_H = (
    "ibm-granite/granite-4.0-h-tiny",
    {
        "num_hidden_layers": 3,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
    },
)
_MAMBA = ("state-spaces/mamba-130m-hf", {"num_hidden_layers": 2})


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_mamba_batch_memcpy_kernel_compiles_on_gpu() -> None:
    _warm_batch_memcpy_kernel(torch.device("cuda"))
    torch.accelerator.synchronize("cuda")


@create_new_process_for_each_test("spawn")
def _check_requests_compile_no_warmed_kernel(
    model: tuple[str, dict],
    speculative_config: dict | None,
    enable_prefix_caching: bool,
) -> None:
    from triton import knobs

    llm = LLM(
        model[0],
        load_format="dummy",
        hf_overrides=model[1],
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.5,
        # These kernels run eagerly; skip compile and graphs for a fast start.
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        enable_prefix_caching=enable_prefix_caching,
        speculative_config=speculative_config,
    )

    compiled: list[str] = []
    monitor_hook = knobs.runtime.jit_post_compile_hook

    def record_compile(**kwargs):
        compiled.append(getattr(kwargs.get("fn"), "name", "<unknown>"))
        if monitor_hook is not None:
            return monitor_hook(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = record_compile
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate("The capital of France is", params)
    # A prompt joining a running decode gets state indices one row in.
    engine = llm.llm_engine
    engine.add_request("decode", "Hello, my name is", params)
    for _ in range(2):
        engine.step()
    engine.add_request("prefill", "The quick brown fox jumps over", params)
    while engine.has_unfinished_requests():
        engine.step()
    assert not _WARMED_KERNELS.intersection(compiled), compiled


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("model", "speculative_config", "enable_prefix_caching"),
    [
        (_LFM2, _NGRAM_SPEC_CONFIG, True),
        (_LFM2, None, True),
        (_LFM2, None, False),
        (_GRANITE_4_H, None, True),
        (_MAMBA, None, True),
    ],
    ids=["align-ngram", "align", "no-prefix-caching", "mamba2-align", "mamba-align"],
)
def test_v1_requests_compile_no_mamba_kernel(
    monkeypatch: pytest.MonkeyPatch,
    model: tuple[str, dict],
    speculative_config: dict | None,
    enable_prefix_caching: bool,
) -> None:
    """Startup warmup must compile the V1 align state copies and Mamba-style
    prefill for the first request and for a prompt joining running decodes,
    or those requests compile them (and die under
    ``jit_monitor_mode="error"``)."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    # Keep the engine core in the child, where the compile hook is installed.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _check_requests_compile_no_warmed_kernel(
        model, speculative_config, enable_prefix_caching
    )
