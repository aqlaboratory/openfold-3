#!/usr/bin/env python
"""Microbenchmark OF3 triangle multiplication variants.

This isolates ``TriangleMultiplication{Outgoing,Incoming}.forward`` and compares
the eager/default path, low-memory fused Triton path, and optionally cuEq. cuEq
first-call and warm timings are reported separately because its gated GEMM
autotune/cache key is length dependent.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import median

import torch

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

from openfold3.core.kernels.triton.fused_trimul import (  # noqa: E402
    gated_dual_gemm_fp32,
    gated_out_gemm_residual_fp32,
    ln_stats_fp32,
    ln_transpose_fp32,
)


def _set_fused(enabled: bool) -> None:
    os.environ["OPENFOLD3_FUSED_TRIMUL"] = "1" if enabled else "0"


def _randomize_observable_output(module) -> None:
    with torch.no_grad():
        for linear in (
            module.linear_a_p,
            module.linear_a_g,
            module.linear_b_p,
            module.linear_b_g,
            module.linear_z,
            module.linear_g,
        ):
            linear.bias = None
        module.linear_z.weight.normal_(0, 0.02)
        module.linear_g.weight.normal_(0, 0.02)


def _bench(fn, *, reps: int, warmup: int) -> tuple[float, float, int]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
            del out
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        wall_times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            wall_times.append((time.perf_counter() - t0) * 1e3)
            del out
        peak = torch.cuda.max_memory_allocated()

    return median(sorted(wall_times)), wall_times[0], peak - base


def _time_stage(fn, *, reps: int, warmup: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
            del out
        torch.cuda.synchronize()

        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            del out
    return median(sorted(times))


def _stage_split(module, z, mask, *, reps: int, warmup: int) -> list[dict]:
    B, N, _, c_z = z.shape
    c_hidden = module.linear_a_p.weight.shape[0]
    z_2d = z.reshape(-1, c_z)
    mask_flat = mask.reshape(-1)
    ln_in = module.layer_norm_in
    ln_out = module.layer_norm_out

    wp = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], dim=0)
    wg = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], dim=0)
    with torch.inference_mode():
        ln_stats = ln_stats_fp32(z_2d, ln_in.eps)
        ab = gated_dual_gemm_fp32(
            z_2d,
            wp,
            wg,
            mask_flat,
            ln_weight=ln_in.weight,
            ln_bias=ln_in.bias,
            ln_stats=ln_stats,
            eps=ln_in.eps,
            output_dtype=None,
        )
        a = ab[:c_hidden].view(c_hidden, B, N, N)
        b = ab[c_hidden:].view(c_hidden, B, N, N)
        if module._outgoing:
            x = torch.einsum("cbik,cbjk->cbij", a, b)
        else:
            x = torch.einsum("cbki,cbkj->cbij", a, b)
        x_dm = x.reshape(c_hidden, B * N * N)
        x_out_2d = ln_transpose_fp32(x_dm, ln_out.weight, ln_out.bias, ln_out.eps)

    def pack_weights():
        return (
            torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], dim=0),
            torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], dim=0),
        )

    def input_ln_stats():
        return ln_stats_fp32(z_2d, ln_in.eps)

    def dual_gemm():
        return gated_dual_gemm_fp32(
            z_2d,
            wp,
            wg,
            mask_flat,
            ln_weight=ln_in.weight,
            ln_bias=ln_in.bias,
            ln_stats=ln_stats,
            eps=ln_in.eps,
            output_dtype=None,
        )

    def einsum_contract():
        if module._outgoing:
            return torch.einsum("cbik,cbjk->cbij", a, b)
        return torch.einsum("cbki,cbkj->cbij", a, b)

    def bmm_contract():
        a2 = a.reshape(c_hidden * B, N, N)
        b2 = b.reshape(c_hidden * B, N, N)
        if module._outgoing:
            return torch.bmm(a2, b2.transpose(-1, -2)).view(c_hidden, B, N, N)
        return torch.bmm(a2.transpose(-1, -2), b2).view(c_hidden, B, N, N)

    def out_ln():
        return ln_transpose_fp32(x_dm, ln_out.weight, ln_out.bias, ln_out.eps)

    def legacy_output():
        return gated_out_gemm_residual_fp32(
            z_2d,
            x_out_2d,
            module.linear_g.weight,
            module.linear_z.weight,
            None,
            ln_weight=ln_in.weight,
            ln_bias=ln_in.bias,
            ln_stats=ln_stats,
            eps=ln_in.eps,
        )

    stages = [
        ("pack_weights", pack_weights),
        ("input_ln_stats", input_ln_stats),
        ("dual_gemm", dual_gemm),
        ("einsum_contract", einsum_contract),
        ("bmm_contract", bmm_contract),
        ("out_ln_transpose", out_ln),
        ("legacy_output_gemm", legacy_output),
    ]
    rows = []
    for label, fn in stages:
        rows.append(
            {
                "stage": label,
                "median_ms": _time_stage(fn, reps=reps, warmup=warmup),
            }
        )
    return rows


def _profile_once(label: str, fn) -> None:
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        out = fn()
        del out
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            out = fn()
            del out
            torch.cuda.synchronize()

    print(f"\n=== CUDA profile: {label} ===")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15,
            top_level_events_only=False,
        )
    )


def _make_case(n: int, *, outgoing: bool, dtype: torch.dtype):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    torch.manual_seed(2026 + n + int(outgoing))
    module = cls(128, 128).cuda().eval().to(dtype=dtype)
    _randomize_observable_output(module)
    z = torch.randn(1, n, n, 128, device="cuda", dtype=dtype) * 0.1
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
    return module, z, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[590, 737, 1264])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Diagnostic only. Default disables TF32 for full-fp32 comparison.",
    )
    parser.add_argument("--profile-n", type=int, default=None)
    parser.add_argument("--stage-split", action="store_true")
    parser.add_argument(
        "--include-cueq",
        action="store_true",
        help="Also benchmark cuEq trimul; first call includes any length-keyed setup.",
    )
    parser.add_argument(
        "--chunk-cap",
        type=int,
        nargs="+",
        default=None,
        help="Add fused_chunked variants with TRIMUL_CHUNK_CAP set.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    variants = [
        ("eager_default", False, False, None),
        ("fused", True, False, None),
    ]
    if args.include_cueq:
        variants.append(("cueq", False, True, None))
    if args.chunk_cap:
        for cap in args.chunk_cap:
            variants.append((f"fused_chunk{cap}", True, False, cap))

    results = []
    header = (
        f"{'variant':<22}{'dir':>6}{'N':>7}{'median ms':>12}"
        f"{'first ms':>11}{'peak U':>10}{'peak MiB':>12}{'tf32':>7}"
    )
    print(header)
    print("-" * len(header))
    try:
        for n in args.n:
            for outgoing in (True, False):
                direction = "out" if outgoing else "in"
                module, z, mask = _make_case(n, outgoing=outgoing, dtype=dtype)
                u_bytes = n * n * 128 * z.element_size()

                for label, fused, use_cueq, chunk_cap in variants:
                    _set_fused(fused)
                    if chunk_cap is not None:
                        os.environ["OPENFOLD3_TRIMUL_CHUNK_CAP"] = str(chunk_cap)
                    else:
                        os.environ.pop("OPENFOLD3_TRIMUL_CHUNK_CAP", None)

                    def run(module=module, z=z, mask=mask, use_cueq=use_cueq):
                        return module(
                            z.clone(),
                            mask=mask,
                            inplace_safe=False,
                            use_cueq_triangle_kernels=use_cueq,
                        )

                    ms, first_ms, peak = _bench(
                        run, reps=args.reps, warmup=args.warmup
                    )
                    row = {
                        "variant": label,
                        "direction": direction,
                        "N": n,
                        "dtype": args.dtype,
                        "tf32": torch.backends.cuda.matmul.allow_tf32,
                        "median_ms": ms,
                        "first_ms": first_ms,
                        "peak_transient_bytes": peak,
                        "peak_transient_mib": peak / 1024**2,
                        "peak_transient_U": peak / u_bytes,
                    }
                    results.append(row)
                    print(
                        f"{label:<22}{direction:>6}{n:>7}{ms:>12.3f}"
                        f"{first_ms:>11.3f}{row['peak_transient_U']:>10.2f}"
                        f"{row['peak_transient_mib']:>12.1f}"
                        f"{str(row['tf32']):>7}"
                    )

                    if args.profile_n == n and outgoing:
                        _profile_once(f"{label} N={n} {direction}", run)

                if args.stage_split:
                    _set_fused(True)
                    stage_rows = _stage_split(
                        module, z, mask, reps=args.reps, warmup=args.warmup
                    )
                    print(f"\nStage split: N={n} dir={direction}")
                    for row in stage_rows:
                        row.update(
                            {
                                "variant": "fused_stage",
                                "direction": direction,
                                "N": n,
                                "dtype": args.dtype,
                                "tf32": torch.backends.cuda.matmul.allow_tf32,
                            }
                        )
                        results.append(row)
                        print(f"  {row['stage']:<22}{row['median_ms']:>10.3f} ms")
                print()

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results, indent=2))
            print(f"Saved {args.output_json}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    main()
