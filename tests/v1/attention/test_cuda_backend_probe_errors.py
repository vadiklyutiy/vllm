# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CUDA attention backend selection and probe error handling.

Environment-shaped probe failures (missing packages, unreadable caches,
broken driver installs) must mark the backend unavailable rather than
terminating engine init; programming errors must still propagate.
See https://github.com/vllm-project/vllm/issues/51658.
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-specific tests"
)

SELECTOR_CONFIG = AttentionSelectorConfig(
    head_size=64,
    dtype=torch.float16,
    kv_cache_dtype=None,
    block_size=16,
)

SM90 = DeviceCapability(major=9, minor=0)


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("flashinfer is not installed"),
        PermissionError(13, "Permission denied", "/root/.cache/flashinfer"),
        OSError("libcuda.so.1: cannot open shared object file"),
    ],
)
def test_get_valid_backends_records_environment_failure(exc):
    with patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert valid == []
    assert invalid_reasons
    for _priority, reasons in invalid_reasons.values():
        assert reasons == [f"{type(exc).__name__}: {exc}"]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA error: no kernel image is available"),
        AttributeError("module has no attribute 'get_builder_cls'"),
        KeyboardInterrupt(),
    ],
)
def test_get_valid_backends_propagates_unexpected_errors(exc):
    with (
        patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc),
        pytest.raises(type(exc)),
    ):
        CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )


def test_get_valid_backends_keeps_probing_after_failure():
    healthy = MagicMock()
    healthy.validate_configuration.return_value = []
    side_effects = [OSError("probe failed")] + [healthy] * 32

    with patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=side_effects):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert len(invalid_reasons) == 1
    assert valid


def test_sm90_nope_mla_prefers_flashinfer_without_changing_rope_order():
    backend_cls = MagicMock()
    backend_cls.validate_configuration.return_value = []
    sparse_backends = {
        AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
    }

    def sparse_order(head_size: int) -> list[AttentionBackendEnum]:
        config = SELECTOR_CONFIG._replace(
            head_size=head_size,
            use_mla=True,
            use_sparse=True,
        )
        with patch(
            "vllm.platforms.cuda._get_attn_backend_class",
            return_value=backend_cls,
        ):
            valid, _ = CudaPlatform.get_valid_backends(
                device_capability=SM90,
                attn_selector_config=config,
                num_heads=32,
            )
        return [
            candidate.backend
            for candidate in valid
            if candidate.backend in sparse_backends
        ]

    assert sparse_order(512) == [
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
        AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE,
    ]
    assert sparse_order(576) == [
        AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
    ]


def test_selected_backend_probe_failure_raises_value_error_with_cause():
    exc = OSError("libcuda.so.1: cannot open shared object file")
    with (
        patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc),
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        pytest.raises(ValueError, match="OSError") as excinfo,
    ):
        CudaPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.FLASH_ATTN,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert excinfo.value.__cause__ is exc


def test_llm_generates_without_fa_extensions():
    """A CUDA build without _vllm_fa2_C/_vllm_fa3_C must start an engine on a
    backend it can run. See https://github.com/vllm-project/vllm/issues/63267."""
    code = (
        "import sys\n"
        "sys.modules['vllm.vllm_flash_attn._vllm_fa2_C'] = None\n"
        "sys.modules['vllm.vllm_flash_attn._vllm_fa3_C'] = None\n"
        "from vllm import LLM, SamplingParams\n"
        "llm = LLM(model='facebook/opt-125m', enforce_eager=True,\n"
        "          gpu_memory_utilization=0.3)\n"
        "params = SamplingParams(temperature=0, max_tokens=8)\n"
        "out = llm.generate(['Hello, my name is'], params)\n"
        "assert out[0].outputs[0].token_ids\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    )
    assert result.returncode == 0, result.stderr


# On SM90 an FA4-only build resolves to the missing FA2.
@pytest.mark.parametrize("fa_versions", [(), (4,)], ids=["no_fa", "fa4_only"])
def test_flash_attn_not_valid_without_usable_fa_version(fa_versions):
    with (
        patch(
            "vllm.vllm_flash_attn.flash_attn_interface.is_fa_version_supported",
            side_effect=lambda v: v in fa_versions,
        ),
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
    ):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert AttentionBackendEnum.FLASH_ATTN in invalid_reasons
    assert valid


@pytest.mark.parametrize("fa_versions", [(), (4,)], ids=["no_fa", "fa4_only"])
def test_vit_skips_flash_attn_without_usable_fa_version(fa_versions):
    with (
        patch(
            "vllm.vllm_flash_attn.flash_attn_interface.is_fa_version_supported",
            side_effect=lambda v: v in fa_versions,
        ),
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
    ):
        backend = CudaPlatform.get_vit_attn_backend(head_size=64, dtype=torch.bfloat16)
    assert backend != AttentionBackendEnum.FLASH_ATTN


def test_flash_attn_not_valid_when_head_size_resolves_to_missing_fa2():
    """FA4-only build on SM100: FA4 has no head_size=192 kernel, so the
    layer resolves to the missing FA2."""
    sm100 = DeviceCapability(major=10, minor=0)
    with (
        patch(
            "vllm.vllm_flash_attn.flash_attn_interface.is_fa_version_supported",
            side_effect=lambda v: v == 4,
        ),
        patch.object(CudaPlatform, "get_device_capability", return_value=sm100),
    ):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=sm100,
            attn_selector_config=SELECTOR_CONFIG._replace(head_size=192),
            num_heads=32,
        )
    assert AttentionBackendEnum.FLASH_ATTN in invalid_reasons
    assert valid
