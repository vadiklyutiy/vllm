# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


@torch.compiler.assume_constant_result
def fp32_dot_allow_tf32() -> bool:
    """``tl.dot(..., allow_tf32=...)`` for fp32 operands that matches PyTorch's
    fp32 matmul precision (set from VLLM_FLOAT32_MATMUL_PRECISION).

    ``True`` keeps Triton's default (TF32 where the target supports it);
    ``False`` gives IEEE. ``TRITON_F32_DEFAULT`` overrides both.
    """
    return torch.backends.cuda.matmul.fp32_precision == "tf32"
