# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""chunk_scaled_dot_kkt_fwd under each of its autotune configs."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
    chunk_scaled_dot_kkt_fwd_kernel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU"
)

BT = 64
# The kernel is wrapped as heuristics -> autotuner -> jit function.
TUNER = chunk_scaled_dot_kkt_fwd_kernel.fn


@pytest.mark.parametrize("config", TUNER.configs, ids=str)
@torch.inference_mode()
def test_chunk_scaled_dot_kkt_fwd_config(config, monkeypatch):
    """Every autotune config gives beta_t * exp(g_t - g_s) * <k_t, k_s>.

    With beta in k's dtype and BK < K, the configs with num_warps >= 4 raced on
    shared memory on sm90 and gave wrong, run-to-run different A.
    """
    monkeypatch.setattr(TUNER, "configs", [config])
    T, H, K = 1024, 8, 128
    N = T // BT

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(1, T, H, K, device=device, generator=gen)
    k = F.normalize(k, dim=-1).bfloat16()
    beta = torch.rand(1, T, H, device=device, generator=gen).bfloat16()
    # Chunk-local cumulative log decay, as chunk_local_cumsum produces it.
    g = torch.rand(1, N, BT, H, device=device, generator=gen) * -0.05
    g = g.cumsum(2).view(1, T, H)

    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=BT)

    # [N, H, BT, ...] per chunk.
    k_c = k.double().view(N, BT, H, K).transpose(1, 2)
    g_c = g.double().view(N, BT, H).transpose(1, 2)
    beta_c = beta.double().view(N, BT, H).transpose(1, 2)
    ref = (k_c @ k_c.transpose(-1, -2)) * beta_c[..., None]
    ref = (ref * torch.exp(g_c[..., None] - g_c[..., None, :])).tril(-1)
    ref = ref.transpose(1, 2).reshape(1, T, H, BT).float()
    torch.testing.assert_close(A, ref)
