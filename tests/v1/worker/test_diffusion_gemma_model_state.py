# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma with use_bidirectional_attention="vision": as in Gemma4,
image tokens are bidirectional on sliding layers and causal on full layers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
    DiffusionGemmaModelState,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import AttentionGroup

BLOCK_SIZE = 16
SLIDING_LAYER = "model.layers.0.self_attn.attn"
FULL_LAYER = "model.layers.5.self_attn.attn"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA backends")
@pytest.mark.parametrize("backend", ["TRITON_ATTN", "FLASH_ATTN"])
def test_image_ranges_reach_sliding_layers_only(backend: str) -> None:
    device = torch.device("cuda:0")
    vllm_config = create_vllm_config(
        model_name="google/diffusiongemma-26B-A4B-it",
        max_model_len=8192,
        block_size=BLOCK_SIZE,
    )
    encoder_cache = EncoderCache()
    encoder_cache.add_request(
        "img",
        [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="img0",
                mm_position=PlaceholderRange(offset=5, length=256),
            )
        ],
    )
    encoder_cache.add_request("txt", [])

    state = object.__new__(DiffusionGemmaModelState)
    state.supports_mm_inputs = True
    state.encoder_cache = encoder_cache
    state.model_config = vllm_config.model_config
    state._causal_buf = torch.zeros(2, dtype=torch.int32, device=device)
    state.diffusion_states = SimpleNamespace(
        is_encoder_phase=torch.ones(2, dtype=torch.bool, device=device)
    )

    query_lens = [274, 16]
    num_tokens = sum(query_lens)
    query_start_loc = np.array([0, 274, 290], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=num_tokens,
        req_ids=["img", "txt"],
        idx_mapping=torch.arange(2, device=device),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        seq_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(query_lens, dtype=torch.int32),
    )

    kv_specs = {
        SLIDING_LAYER: SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=8,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
        ),
        FULL_LAYER: FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=8,
            head_size=256,
            dtype=torch.bfloat16,
        ),
    }
    with set_current_vllm_config(vllm_config):
        attn_groups = []
        for group_id, (layer, kv_spec) in enumerate(kv_specs.items()):
            group = AttentionGroup(
                backend=AttentionBackendEnum[backend].get_class(),
                layer_names=[layer],
                kv_cache_spec=kv_spec,
                kv_cache_group_id=group_id,
            )
            group.create_metadata_builders(vllm_config, device)
            attn_groups.append([group])
        block_table = torch.zeros(
            2, 8192 // BLOCK_SIZE, dtype=torch.int32, device=device
        )
        metadata = state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=(block_table, block_table),
            slot_mappings=torch.arange(num_tokens, device=device).repeat(2, 1),
            attn_groups=attn_groups,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None, None]),
        )

    model = object.__new__(
        DiffusionGemmaForConditionalGeneration  # type: ignore[type-abstract]  # protocol attrs unset
    )
    nn.Module.__init__(model)
    model._full_attn_layer_idxs = frozenset({5})
    model.model = lambda **kwargs: kwargs["inputs_embeds"]
    with set_forward_context(metadata, vllm_config):
        model(
            input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
            positions=torch.arange(num_tokens, device=device),
        )

    sliding, full = metadata[SLIDING_LAYER], metadata[FULL_LAYER]
    if backend == "TRITON_ATTN":
        assert sliding.mm_prefix_range == {0: [(5, 260)], 1: []}
        assert sliding.mm_prefix_range_tensor is not None
    else:
        assert sliding.mm_prefix_query_range_tensor is not None
    for field in (
        "mm_prefix_range",
        "mm_prefix_range_tensor",
        "mm_prefix_query_range_tensor",
    ):
        assert getattr(full, field, None) is None
