# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.platforms
from vllm.config import CompilationMode
from vllm.distributed import parallel_state
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v32.amd import model as deepseek_v32_amd_model
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.dots3_note.nvidia import model as dots3_note_model

HIDDEN_SIZE = 4
INDEX_TOPK = 3
NUM_TOKENS = 5
STALE_INDEX = -7


class _IndexerLayer(nn.Module):
    """Writes the batch's top-k into the shared buffer, as an indexer does."""

    def __init__(self, topk_indices_buffer: torch.Tensor, topk_indices: torch.Tensor):
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_indices = topk_indices

    def forward(self, positions, hidden_states, residual, *args):
        self.topk_indices_buffer[: positions.shape[0]] = self.topk_indices
        return hidden_states, hidden_states if residual is None else residual


class _SharedTopkLayer(nn.Module):
    """Records the top-k a layer without an indexer attends with."""

    def __init__(self, topk_indices_buffer: torch.Tensor):
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer
        self.seen_topk_indices: torch.Tensor | None = None

    def forward(self, positions, hidden_states, residual, *args):
        self.seen_topk_indices = self.topk_indices_buffer[: positions.shape[0]].clone()
        return hidden_states, residual


@pytest.mark.parametrize(
    ("model_module", "model_cls_name", "layer_cls_name", "embedding_name"),
    [
        pytest.param(
            deepseek_v32_model,
            "DeepseekV32Model",
            "DeepseekV32DecoderLayer",
            "make_input_embedding",
            id="cuda",
        ),
        pytest.param(
            deepseek_v32_amd_model,
            "DeepseekV32Model",
            "DeepseekV32DecoderLayer",
            "VocabParallelEmbedding",
            id="rocm",
        ),
        pytest.param(
            deepseek_v2,
            "DeepseekV2Model",
            "DeepseekV2DecoderLayer",
            "VocabParallelEmbedding",
            id="generic",
        ),
        pytest.param(
            dots3_note_model,
            "Dots3NoteModel",
            "Dots3NoteDecoderLayer",
            "VocabParallelEmbedding",
            id="dots3_note",
        ),
    ],
)
def test_stage_without_indexer_reuses_previous_stage_topk(
    monkeypatch, model_module, model_cls_name, layer_cls_name, embedding_name
):
    """A stage whose first layer has no indexer must attend with the top-k
    computed by the previous stage, not with stale rows of its own buffer."""
    topk_indices = torch.arange(NUM_TOKENS * INDEX_TOPK, dtype=torch.int32).view(
        NUM_TOKENS, INDEX_TOPK
    )
    config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=HIDDEN_SIZE,
        index_topk=INDEX_TOPK,
        vocab_size=8,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=None,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2 * NUM_TOKENS, max_num_seqs=NUM_TOKENS
        ),
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            pipeline_parallel_size=2,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
    )
    cpu = SimpleNamespace(device_type="cpu")
    monkeypatch.setattr(vllm.platforms, "current_platform", cpu, raising=False)
    monkeypatch.setattr(model_module, "current_platform", cpu, raising=False)
    monkeypatch.setattr(model_module, embedding_name, lambda *a, **kw: PPMissingLayer())
    model_cls = getattr(model_module, model_cls_name)

    def set_pp_rank(rank: int) -> None:
        pp_group = SimpleNamespace(
            rank_in_group=rank,
            world_size=2,
            is_first_rank=rank == 0,
            is_last_rank=False,
        )
        for module in {parallel_state, deepseek_v32_model, model_module}:
            monkeypatch.setattr(module, "get_pp_group", lambda: pp_group)

    def make_stage(rank: int, make_layer) -> nn.Module:
        # The real constructor runs; only the decoder layers, whose sparse MLA
        # attention needs GPU backends, are stand-ins.
        set_pp_rank(rank)
        monkeypatch.setattr(
            model_module,
            layer_cls_name,
            lambda vllm_config, prefix, topk_indices_buffer, **kw: make_layer(
                topk_indices_buffer
            ),
        )
        stage = model_cls(vllm_config=vllm_config, prefix="")
        stage.layers[rank].topk_indices_buffer.fill_(STALE_INDEX)
        return stage

    first_stage = make_stage(0, lambda buffer: _IndexerLayer(buffer, topk_indices))
    second_stage = make_stage(1, _SharedTopkLayer)
    positions = torch.arange(NUM_TOKENS)

    set_pp_rank(0)
    sent = first_stage(
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=torch.ones(NUM_TOKENS, HIDDEN_SIZE),
    )
    # The model runners receive into these buffers, key by key.
    received = second_stage.make_empty_intermediate_tensors(
        batch_size=NUM_TOKENS, dtype=torch.float32, device=torch.device("cpu")
    )
    for key, buffer in received.items():
        buffer.copy_(sent[key])
    set_pp_rank(1)
    second_stage(input_ids=None, positions=positions, intermediate_tensors=received)

    torch.testing.assert_close(second_stage.layers[1].seen_topk_indices, topk_indices)
    # copy_ casts, so check the receive buffers' dtypes directly.
    assert {k: (v.dtype, v.shape) for k, v in received.items()} == {
        k: (v.dtype, v.shape) for k, v in sent.items()
    }
