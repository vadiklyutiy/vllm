# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import shutil

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

ERROR_CASES = [
    (
        "test_rank",
        {"r": 1024},
        "is greater than max_lora_rank",
    ),
    ("test_dora", {"use_dora": True}, "does not yet support DoRA"),
    (
        "test_modules_to_save",
        {"modules_to_save": ["lm_head"]},
        "Unsupported modules_to_save",
    ),
    ("test_rank_zero", {"r": 0}, "must be a positive integer"),
    ("test_rank_negative", {"r": -8}, "must be a positive integer"),
]


def test_peft_helper_pass(llama32_lora_files, tmp_path):
    peft_helper = PEFTHelper.from_local_dir(
        llama32_lora_files, max_position_embeddings=4096
    )
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    peft_helper.validate_legal(lora_config)
    assert peft_helper.r == 8
    assert peft_helper.lora_alpha == 32
    target_modules = sorted(peft_helper.target_modules)

    assert target_modules == [
        "down_proj",
        "embed_tokens",
        "gate_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert peft_helper.vllm_max_position_embeddings == 4096

    # test RSLoRA
    rslora_config = dict(use_rslora=True)
    test_dir = tmp_path / "test_rslora"
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(rslora_config)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)

    peft_helper = PEFTHelper.from_local_dir(test_dir, max_position_embeddings=4096)
    peft_helper.validate_legal(lora_config)
    scaling = peft_helper.lora_alpha / math.sqrt(peft_helper.r)
    assert abs(peft_helper.vllm_lora_scaling_factor - scaling) < 1e-3


@pytest.mark.parametrize("test_name,config_change,expected_error", ERROR_CASES)
def test_peft_helper_error(
    llama32_lora_files,
    tmp_path,
    test_name: str,
    config_change: dict,
    expected_error: str,
):
    test_dir = tmp_path / test_name
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(config_change)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    # Test loading the adapter
    with pytest.raises(ValueError, match=expected_error):
        PEFTHelper.from_local_dir(
            test_dir, max_position_embeddings=4096
        ).validate_legal(lora_config)


@pytest.mark.parametrize("bad_rank", [0, -1, -8])
def test_peft_helper_invalid_rank_direct(bad_rank: int):
    """Regression test: constructing a PEFTHelper with a non-positive rank
    must raise a clear ValueError instead of crashing with an unrelated
    ZeroDivisionError (r=0) or silently succeeding with a sign-flipped
    scaling factor that validate_legal() never catches (r<0, since its only
    rank check is the upper bound against max_lora_rank).

    Network-free: constructs PEFTHelper directly rather than going through
    from_local_dir(), which needs an on-disk adapter_config.json.
    """
    with pytest.raises(ValueError, match="must be a positive integer"):
        PEFTHelper(r=bad_rank, lora_alpha=16, target_modules=["q_proj"])


@pytest.mark.skip_global_cleanup
def test_rank_and_alpha_pattern_scaling():
    """Modules matched by rank_pattern/alpha_pattern get PEFT's per-module
    alpha / r scaling."""
    peft_helper = PEFTHelper.from_dict(
        {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "rank_pattern": {"q_proj": 4},
            "alpha_pattern": {"layers.1.self_attn.v_proj": 8},
        }
    )
    tensors = {}
    for layer in (0, 1):
        for module, rank in (("q_proj", 4), ("v_proj", 16)):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros(rank, 8)
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros(8, rank)

    lora_model = LoRAModel.from_lora_tensors(1, tensors, peft_helper, device="cpu")

    assert {name: lora.scaling for name, lora in lora_model.loras.items()} == {
        "model.layers.0.self_attn.q_proj": 32 / 4,
        "model.layers.0.self_attn.v_proj": 32 / 16,
        "model.layers.1.self_attn.q_proj": 32 / 4,
        "model.layers.1.self_attn.v_proj": 8 / 16,
    }
