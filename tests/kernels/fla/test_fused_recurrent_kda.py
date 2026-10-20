# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="KDA Triton kernels require a CUDA-alike device.",
)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """fp32 reference for one sequence: ``[T, H, K]`` inputs, ``[H, V, K]``
    state; mirrors the kernel's q/k l2norm and default scale.
    """
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * g[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out


@pytest.mark.parametrize("head_dim", [64, 100])
@torch.inference_mode()
def test_fused_recurrent_kda_ignores_memory_past_gate(head_dim: int):
    """The kernel covers ``next_power_of_2(K)`` key lanes. The padding lanes
    must not read the gate: for the last token they would read past the end
    of ``g``, so ``g`` is placed right before NaNs.
    """
    torch.manual_seed(0)
    T, H, K = 5, 2, head_dim
    q = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    buf = torch.full((T * H * K + K,), float("nan"), device=DEVICE)
    g = buf[: T * H * K].view(1, T, H, K)
    g.copy_(-torch.rand_like(g))
    beta = torch.rand(1, T, H, device=DEVICE, dtype=torch.bfloat16)
    state = torch.randn(1, H, K, K, device=DEVICE)

    out, _ = fused_recurrent_kda(
        q, k, v, g, beta, initial_state=state.clone(), inplace_final_state=False
    )

    expected = naive_recurrent_kda(q[0], k[0], v[0], g[0], beta[0], state[0])
    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=1e-2)
