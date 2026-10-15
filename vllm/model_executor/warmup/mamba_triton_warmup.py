# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Mamba-style Triton kernels shared across GDN / Mamba / KDA models."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _has_mamba_style_cache(runner: "GPUModelRunner") -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    for group in runner.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, MambaSpec):
            return True
    return False


def _warm_batch_memcpy_kernel(device: torch.device) -> None:
    """Warm the Mamba prefix-cache state copy specialization."""
    from vllm.v1.worker.mamba_utils import batch_memcpy

    src = torch.empty(1024, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    batch_memcpy(
        torch.tensor([src.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([dst.data_ptr()], dtype=torch.uint64, device=device),
        # Keep this int32: Triton's compile key includes pointer element dtypes,
        # and the production prefix-cache path passes an int32 sizes tensor.
        torch.tensor([src.numel()], dtype=torch.int32, device=device),
    )
    logger.info("Warmed Mamba batch_memcpy_kernel.")


def _warm_align_state_copy_kernels(runner: "GPUModelRunner") -> None:
    """Warm the fused align state copies run with speculative decoding."""
    ctx = runner._get_mamba_bufs().postprocess_align
    if ctx is None:
        return
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.precopy_src_col_buf is not None
    assert ctx.precopy_token_bias_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    ctx.initialize_from_forward_context(
        runner.kv_cache_config,
        runner.compilation_config.static_forward_context,
        runner._get_mamba_state_copy_funcs(),
        [
            runner.input_batch.block_table[gid].get_device_tensor(1)
            for gid in ctx.mamba_group_ids
        ],
    )
    # A new request scheduling its first token needs no state copy, so both
    # launches only compile.
    for buf, value in (
        (ctx.mamba_state_idx_buf, 0),
        (ctx.precopy_src_col_buf, -1),
        (ctx.num_scheduled_tokens_buf, 1),
        (ctx.num_computed_tokens_buf, 0),
        (ctx.num_draft_tokens_buf, 0),
    ):
        buf.np[0] = value
        buf.copy_to_gpu(1)
    ctx.run_fused_precopy(
        num_reqs=1,
        state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        src_col_gpu=ctx.precopy_src_col_buf.gpu,
        token_bias_gpu=ctx.precopy_token_bias_buf.gpu,
        idx_mapping=None,
    )
    ctx.run_fused_postprocess(
        num_reqs=1,
        num_accepted_tokens_gpu=runner.num_accepted_tokens.gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )
    logger.info("Warmed Mamba align state copy kernels.")


@torch.inference_mode()
def mamba_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm the prefix-cache memcpy and, on V1, the align state copies."""
    device = runner.device
    if device.type != "cuda":
        return
    if not _has_mamba_style_cache(runner):
        return
    _warm_batch_memcpy_kernel(device)
    # The V2 runner warms these paths by running real steps.
    if runner.vllm_config.use_v2_model_runner:
        return
    if runner.cache_config.mamba_cache_mode == "align":
        _warm_align_state_copy_kernels(runner)
