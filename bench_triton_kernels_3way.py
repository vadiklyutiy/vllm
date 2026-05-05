# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""3-way comparison: _fwd (orig) vs _fwd (query_len=1 constexpr) vs _packed_decode.

Same shapes (Qwen3.5-35B-A3B GDN dims), bf16 q/k/v + bf16/fp32 SSM state,
cold L2, no CUDA graph.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, "/home/vgimpelson/1/flashinfer")

import flashinfer  # noqa: E402
import torch  # noqa: E402
from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: E402

from _fwd_kernel_orig import call_fwd_orig  # noqa: E402
from vllm.model_executor.layers.fla.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_packed_decode,
)

H, HV, K, V = 16, 32, 128, 128
CONV_DIM = 2 * K * H + V * HV
SCALE = K**-0.5


def make_fwd_kwargs(B: int, state_dtype: torch.dtype):
    dev = torch.device("cuda:0")
    T = 1
    total = B * T
    torch.manual_seed(B)
    q = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=dev) * 0.05
    k = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=dev) * 0.05
    v = torch.randn(1, total, HV, V, dtype=torch.bfloat16, device=dev) * 0.05
    g = torch.randn(1, total, HV, dtype=torch.float32, device=dev) * 0.1
    beta = torch.randn(1, total, HV, dtype=torch.bfloat16, device=dev) * 0.05
    pool = max(8192, total)
    initial_state = torch.zeros(pool, HV, V, K, dtype=state_dtype, device=dev)
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * T
    ssm_state_indices = torch.arange(0, B, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=SCALE,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def make_packed_kwargs(B: int, state_dtype: torch.dtype):
    dev = torch.device("cuda:0")
    torch.manual_seed(B)
    mixed_qkv = torch.randn(B, CONV_DIM, dtype=torch.bfloat16, device=dev) * 0.05
    a = torch.randn(B, HV, dtype=torch.bfloat16, device=dev) * 0.05
    b = torch.randn(B, HV, dtype=torch.bfloat16, device=dev) * 0.05
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=dev) * 0.1
    pool = max(8192, B)
    initial_state = torch.zeros(pool, HV, V, K, dtype=state_dtype, device=dev)
    out = torch.empty(B, 1, HV, V, dtype=torch.bfloat16, device=dev)
    ssm_state_indices = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    return dict(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=SCALE,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )


def _stats(times_ms):
    us = [t * 1000.0 for t in times_ms]
    n = len(us)
    med = statistics.median(us)
    sd = statistics.stdev(us) if n > 1 else 0.0
    return med, sd / (n**0.5), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    )
    p.add_argument("--state-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    if not torch.accelerator.is_available():
        sys.exit("Accelerator required.")

    sd = torch.bfloat16 if args.state_dtype == "bfloat16" else torch.float32

    print(
        f"flashinfer={flashinfer.__file__} v={flashinfer.__version__}", file=sys.stderr
    )
    print(f"H={H} HV={HV} K={K} V={V}, state_dtype={args.state_dtype}", file=sys.stderr)

    grid_y_max = 65535
    triton_max_b = grid_y_max // HV

    for B in args.batches:
        if triton_max_b < B:
            continue
        kw_fwd = make_fwd_kwargs(B, sd)
        kw_pack = make_packed_kwargs(B, sd)
        for _ in range(3):
            call_fwd_orig(**kw_fwd)
            fused_recurrent_gated_delta_rule_fwd(**kw_fwd, query_len=1)
            fused_recurrent_gated_delta_rule_packed_decode(**kw_pack)
    torch.accelerator.synchronize()

    rows = []
    for B in args.batches:
        if triton_max_b < B:
            rows.append((B, None, None, None))
            continue
        kw_fwd = make_fwd_kwargs(B, sd)
        kw_pack = make_packed_kwargs(B, sd)
        t_orig = bench_gpu_time_with_cupti(
            fn=lambda **kw: call_fwd_orig(**kw),
            input_kwargs=kw_fwd,
            dry_run_time_ms=25,
            repeat_time_ms=100,
            cold_l2_cache=True,
            use_cuda_graph=False,
            sleep_after_run=False,
        )
        kw_fwd_new = dict(kw_fwd, query_len=1)
        # Drop inplace_final_state from kwargs; new wrapper requires it positional
        t_new = bench_gpu_time_with_cupti(
            fn=lambda **kw: fused_recurrent_gated_delta_rule_fwd(**kw),
            input_kwargs=kw_fwd_new,
            dry_run_time_ms=25,
            repeat_time_ms=100,
            cold_l2_cache=True,
            use_cuda_graph=False,
            sleep_after_run=False,
        )
        t_pack = bench_gpu_time_with_cupti(
            fn=lambda **kw: fused_recurrent_gated_delta_rule_packed_decode(**kw),
            input_kwargs=kw_pack,
            dry_run_time_ms=25,
            repeat_time_ms=100,
            cold_l2_cache=True,
            use_cuda_graph=False,
            sleep_after_run=False,
        )
        rows.append((B, _stats(t_orig), _stats(t_new), _stats(t_pack)))

    print()
    print(
        f"### state_dtype={args.state_dtype}, q/k/v=bf16, cold-L2, no CUDA graph, B200"
    )
    print()
    print(
        "| B    | _fwd orig (us)     | _fwd query_len=1 (us) | _packed_decode (us) | "
        "orig/new | new/packed | orig/packed |"
    )
    print(
        "|------|--------------------|-----------------------|---------------------|"
        "----------|------------|-------------|"
    )
    for B, so, sn, sp in rows:
        if so is None:
            print(
                f"| {B:4d} | N/A                | N/A                   | "
                f"N/A                 | N/A      | N/A        | N/A         |"
            )
            continue
        mo, seo, _ = so
        mn, sen, _ = sn
        mp, sep, _ = sp
        on = mo / mn if mn > 0 else float("nan")
        np_ = mn / mp if mp > 0 else float("nan")
        op = mo / mp if mp > 0 else float("nan")
        print(
            f"| {B:4d} | {mo:9.2f} ± {seo:5.2f}  "
            f"| {mn:11.2f} ± {sen:5.2f}    "
            f"| {mp:11.2f} ± {sep:5.2f}   "
            f"| {on:6.3f}× | {np_:6.3f}×    | {op:6.3f}×    |"
        )


if __name__ == "__main__":
    main()
