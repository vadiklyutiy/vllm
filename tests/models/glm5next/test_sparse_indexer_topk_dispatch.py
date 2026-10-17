# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


def _build(indexer_cls, backend: str):
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return indexer_cls(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "aiter", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_rocm(
    backend: str,
) -> None:
    """The AMD kpool indexer must go through the same dispatcher, so the
    AITER decode top-k is reachable from kernel_config."""
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize(
    "configured,full_cudagraph,expected",
    [
        ("auto", False, "aiter"),
        ("auto", True, "auto"),
        ("aiter", True, "aiter"),
        ("per_row", False, "per_row"),
    ],
)
def test_kpool_decode_topk_backend_never_auto_selects_aiter_under_full_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    full_cudagraph: bool,
    expected: str,
) -> None:
    """A FULL cudagraph bakes in one kernel for every replay, so "auto" cannot
    use the context length to pick AITER and must stay on the in-tree kernel.
    Explicit backends keep working, and outside FULL the long-context heuristic
    still narrows to AITER."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd import sparse_indexer

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: True, raising=False
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_indexer_top_k_supported",
        lambda **kwargs: True,
        raising=False,
    )

    assert (
        sparse_indexer._kpool_decode_topk_backend(
            configured,
            num_rows=8,
            max_valid_seq_len=64 * 1024,
            select_k=512,
            index_kpool=4,
            full_cudagraph=full_cudagraph,
        )
        == expected
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only indexer")
def test_kpool_profile_run_reserves_every_ubatch_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profile run is not microbatched, but under DBO microbatch 1 asks for
    the prefill K-gather buffers only after the workspace is locked."""
    from vllm.models.glm5next.nvidia import sparse_indexer
    from vllm.v1.worker import workspace

    device = torch.device("cuda")
    manager = workspace.WorkspaceManager(device, num_ubatches=2)
    ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: ubatch[0])
    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    hidden = torch.zeros((16, 128), device=device)
    sparse_indexer.sparse_attn_indexer_kpool(
        hidden,
        "k",
        torch.empty(0, device=device),
        hidden,
        None,
        None,
        hidden,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2048,
        head_dim=128,
        max_pool_len=4096,
        total_seq_lens=8192,
        topk_indices_buffer=torch.empty((16, 2048), dtype=torch.int32, device=device),
        skip_k_cache_insert=True,
    )
    manager.lock()

    ubatch[0] = 1
    manager.get_simultaneous(
        *sparse_indexer._gather_workspace_shapes(
            8192, 128, current_platform.fp8_dtype(), False
        )
    )
