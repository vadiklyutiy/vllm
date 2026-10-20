# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize(
    "dtype,state_dtype,out_dtype",
    [
        (torch.float16, torch.float32, torch.float32),
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float64, torch.float32),
        (torch.bfloat16, torch.float32, torch.bfloat16),
    ],
)
def test_chunk_gated_delta_rule_state_beyond_fp16_range(
    dtype: torch.dtype, state_dtype: torch.dtype, out_dtype: torch.dtype
):
    """A state larger than fp16's max must not overflow the chunk path."""
    torch.manual_seed(0)
    B, T, H, K, V = 1, 256, 4, 128, 128
    q = torch.randn(B, T, H, K, device=DEVICE)
    k = F.normalize(torch.randn(B, T, H, K, device=DEVICE), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=DEVICE)
    g = torch.full((B, T, H), -0.01, device=DEVICE)
    beta = torch.full((B, T, H), 0.5, device=DEVICE)
    initial_state = (torch.randn(B, H, V, K, device=DEVICE) * 1e6).to(state_dtype)

    ref_o, ref_states = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.float(),
        inplace_final_state=False,
    )
    o, final_state = chunk_gated_delta_rule(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        g=g.to(dtype),
        beta=beta.to(dtype),
        initial_state=initial_state,
        output_final_state=True,
    )

    assert _rel_err(o, ref_o) < 1e-2
    assert _rel_err(final_state, ref_states[-1:]) < 1e-2
    assert o.dtype == out_dtype
