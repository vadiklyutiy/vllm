# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OLMo Hybrid checkpoints store the GDN short convolution as separate
`q_conv1d`/`k_conv1d`/`v_conv1d` weights, which `OlmoHybridModel.hf_to_vllm_mapper`
stacks into one fused `conv1d` parameter. `AutoWeightsLoader` calls a parameter's
weight loader without a shard id, so the fused loader must take it from the
mapped tensor.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    _make_fused_conv1d_weight_loader,
)
from vllm.model_executor.models.olmo_hybrid import OlmoHybridModel
from vllm.model_executor.models.utils import AutoWeightsLoader

KEY_DIM = 4
VALUE_DIM = 8
KERNEL_SIZE = 3


def _model_with_fused_conv1d(tp_size: int, tp_rank: int) -> nn.Module:
    dims = [KEY_DIM, KEY_DIM, VALUE_DIM]
    conv1d = nn.Module()
    conv1d.weight = nn.Parameter(torch.zeros(sum(dims) // tp_size, 1, KERNEL_SIZE))
    conv1d.weight.weight_loader = _make_fused_conv1d_weight_loader(
        dims, tp_size, tp_rank
    )
    linear_attn = nn.Module()
    linear_attn.conv1d = conv1d
    layer = nn.Module()
    layer.linear_attn = linear_attn
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    return model


@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 1)])
def test_split_conv1d_weights_load_into_fused_conv1d(tp_size, tp_rank):
    model = _model_with_fused_conv1d(tp_size, tp_rank)
    q, k, v = (
        torch.randn(dim, 1, KERNEL_SIZE) for dim in (KEY_DIM, KEY_DIM, VALUE_DIM)
    )
    weights = [
        ("layers.0.linear_attn.q_conv1d.weight", q),
        ("layers.0.linear_attn.k_conv1d.weight", k),
        ("layers.0.linear_attn.v_conv1d.weight", v),
    ]

    loaded = AutoWeightsLoader(model).load_weights(
        weights, mapper=OlmoHybridModel.hf_to_vllm_mapper
    )

    assert loaded == {"layers.0.linear_attn.conv1d.weight"}
    expected = torch.cat([w.chunk(tp_size)[tp_rank] for w in (q, k, v)])
    torch.testing.assert_close(model.layers[0].linear_attn.conv1d.weight, expected)
