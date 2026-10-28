# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEncoder global attention: the fused Triton relative-position kernel has
to compile and match the SDPA path it replaces."""

import pytest
import torch

from vllm.model_executor.models.deepencoder import RelPosAttention
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="Triton path needs CUDA/ROCm"
    ),
]

DIM = 768
NUM_HEADS = 12
# 14 * 14 = 196 tokens, so the last query and key blocks are partial.
SIZE = 14


@torch.no_grad()
def test_triton_rel_pos_attention_matches_sdpa():
    torch.manual_seed(0)
    kwargs = dict(
        dim=DIM, num_heads=NUM_HEADS, use_rel_pos=True, input_size=(SIZE, SIZE)
    )
    sdpa_attn = RelPosAttention(**kwargs, use_triton_attention=False)
    sdpa_attn.rel_pos_h.normal_()
    sdpa_attn.rel_pos_w.normal_()
    triton_attn = RelPosAttention(**kwargs, use_triton_attention=True)
    triton_attn.load_state_dict(sdpa_attn.state_dict())
    sdpa_attn = sdpa_attn.to("cuda", torch.bfloat16)
    triton_attn = triton_attn.to("cuda", torch.bfloat16)

    x = torch.randn(2, SIZE, SIZE, DIM, device="cuda", dtype=torch.bfloat16)

    torch.testing.assert_close(triton_attn(x), sdpa_attn(x), atol=2e-2, rtol=2e-2)
