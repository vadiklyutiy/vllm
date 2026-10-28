# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def _run_dummy_batch(
    speculator,
    *,
    max_num_reqs=512,
    is_profile=False,
    skip_attn=False,
):
    captured = {}

    def execute_model(scheduler_output, **kwargs):
        captured["scheduler_output"] = scheduler_output
        raise _StopDummyRun

    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        decode_query_len=1,
        speculator=speculator,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_first_pp_rank=True,
        lora_config=None,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        execute_model=execute_model,
    )

    with pytest.raises(_StopDummyRun):
        GPUModelRunner._dummy_run(
            runner,
            num_tokens=2048,
            skip_attn=skip_attn,
            is_profile=is_profile,
        )

    return captured["scheduler_output"]


def test_default_speculator_query_width_is_one():
    assert _DefaultSpeculator().num_query_per_req == 1


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DSparkSpeculator])
@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, False), (True, True)],
)
def test_dummy_run_caps_parallel_draft_requests(speculator_cls, is_profile, skip_attn):
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(
        speculator,
        is_profile=is_profile,
        skip_attn=skip_attn,
    )
    assert len(scheduler_output.num_scheduled_tokens) == 409
    assert scheduler_output.total_num_scheduled_tokens == 2048


def test_dummy_run_keeps_already_fitting_parallel_draft_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(speculator, max_num_reqs=128)
    assert len(scheduler_output.num_scheduled_tokens) == 128


def test_dummy_run_keeps_default_speculator_batch():
    scheduler_output = _run_dummy_batch(_DefaultSpeculator())
    assert len(scheduler_output.num_scheduled_tokens) == 512


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Adaptive verification needs CUDA graphs."
)
def test_adaptive_verification_starts_below_draft_rows_per_seq(tmp_path):
    """Adaptive verification profiles dummy batches padded to FULL graphs sized
    by max_num_seqs; the draft pass must keep within max_num_batched_tokens."""
    num_speculative_tokens = 7
    max_num_seqs = 256
    max_num_batched_tokens = 256
    assert max_num_seqs * num_speculative_tokens > max_num_batched_tokens

    config = dict(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        hidden_act="silu",
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    draft_config = dict(
        config,
        architectures=["Qwen3DSparkModel"],
        num_hidden_layers=1,
        block_size=num_speculative_tokens,
        num_target_layers=4,
        dflash_config=dict(mask_token_id=0, target_layer_ids=[0, 1, 2]),
        markov_rank=32,
        markov_head_type="vanilla",
        sample_from_anchor=True,
    )
    for name, model_config in (("target", config), ("draft", draft_config)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(model_config))

    llm = LLM(
        model=str(tmp_path / "target"),
        skip_tokenizer_init=True,
        load_format="dummy",
        max_model_len=2048,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.3,
        # Replays varlen decode graphs, which FlashAttention 2 does not support.
        attention_backend="TRITON_ATTN",
        speculative_config=dict(
            method="dspark",
            model=str(tmp_path / "draft"),
            num_speculative_tokens=num_speculative_tokens,
            enable_adaptive_verification=True,
        ),
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])] * 4,
        SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
    )
    assert [len(output.outputs[0].token_ids) for output in outputs] == [16] * 4
