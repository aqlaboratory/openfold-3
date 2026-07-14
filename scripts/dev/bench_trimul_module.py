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
from statistics import median, pstdev

import torch

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

from openfold3.core.kernels.triton.fused_trimul import (  # noqa: E402
    gated_dual_gemm_fp32,
    gated_out_from_dm_residual_fp32,
    gated_out_gemm_residual_fp32,
    ln_stats_fp32,
    ln_transpose_fp32,
)
from openfold3.core.model.primitives.fused_trimul import (  # noqa: E402
    fused_trimul_update,
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


def _cuda_time_call(fn, work: torch.Tensor) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(work)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), out


def _bench(
    prepare,
    fn,
    *,
    reps: int,
    warmup: int,
) -> tuple[float, float, float, int, float]:
    """Return warm median/std, true pre-warm call, transient peak, cell wall.

    ``prepare`` runs and synchronizes outside each timed region. This is
    important for the legacy Triton path, which overwrites its input: reset
    copies should not be charged to that implementation while the other
    backends receive an already-prepared input.
    """
    cell_t0 = time.perf_counter()
    with torch.inference_mode():
        work = prepare()
        torch.cuda.synchronize()
        first_ms, out = _cuda_time_call(fn, work)
        del out, work

        for _ in range(warmup):
            work = prepare()
            torch.cuda.synchronize()
            _, out = _cuda_time_call(fn, work)
            del out, work

        warm_times = []
        for _ in range(reps):
            work = prepare()
            torch.cuda.synchronize()
            elapsed_ms, out = _cuda_time_call(fn, work)
            warm_times.append(elapsed_ms)
            del out, work

        gc.collect()
        torch.cuda.empty_cache()
        work = prepare()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _, out = _cuda_time_call(fn, work)
        peak = torch.cuda.max_memory_allocated()
        del out, work

    cell_wall_s = time.perf_counter() - cell_t0
    return (
        median(sorted(warm_times)),
        pstdev(warm_times) if len(warm_times) > 1 else 0.0,
        first_ms,
        peak - base,
        cell_wall_s,
    )


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


def _subop_split(
    module,
    z,
    mask,
    *,
    row_caps: list[int | None],
    reps: int,
    warmup: int,
) -> list[dict]:
    """Time target-file kernels directly at full and chunk-sized row counts."""
    _, N, _, c_z = z.shape
    c_hidden = module.linear_a_p.weight.shape[0]
    z_2d = z.reshape(-1, c_z)
    mask_flat = mask.reshape(-1)
    ln_in = module.layer_norm_in
    ln_out = module.layer_norm_out
    wp_ab = torch.cat(
        [module.linear_a_p.weight, module.linear_b_p.weight], dim=0,
    )
    wg_ab = torch.cat(
        [module.linear_a_g.weight, module.linear_b_g.weight], dim=0,
    )
    rows = []
    seen_m = set()

    for row_cap in row_caps:
        active_rows = N if row_cap is None else min(row_cap, N)
        M = active_rows * N
        if M in seen_m:
            continue
        seen_m.add(M)
        z_c = z_2d[:M]
        mask_c = mask_flat[:M]
        with torch.inference_mode():
            stats = ln_stats_fp32(z_c, ln_in.eps)
            x_dm = torch.randn(
                c_hidden, M, device=z.device, dtype=z.dtype,
            )
            x_out = ln_transpose_fp32(
                x_dm, ln_out.weight, ln_out.bias, ln_out.eps,
            )
            out_buf = torch.empty_like(z_c)

        def dual_ab(z_c=z_c, mask_c=mask_c, stats=stats):
            return gated_dual_gemm_fp32(
                z_c,
                wp_ab,
                wg_ab,
                mask_c,
                ln_weight=ln_in.weight,
                ln_bias=ln_in.bias,
                ln_stats=stats,
                eps=ln_in.eps,
            )

        def dual_single(z_c=z_c, mask_c=mask_c, stats=stats):
            return gated_dual_gemm_fp32(
                z_c,
                module.linear_a_p.weight,
                module.linear_a_g.weight,
                mask_c,
                ln_weight=ln_in.weight,
                ln_bias=ln_in.bias,
                ln_stats=stats,
                eps=ln_in.eps,
            )

        def out_alloc(z_c=z_c, x_out=x_out, stats=stats):
            return gated_out_gemm_residual_fp32(
                z_c,
                x_out,
                module.linear_g.weight,
                module.linear_z.weight,
                None,
                ln_weight=ln_in.weight,
                ln_bias=ln_in.bias,
                ln_stats=stats,
                eps=ln_in.eps,
            )

        def out_preallocated(
            z_c=z_c, x_out=x_out, stats=stats, out_buf=out_buf,
        ):
            return gated_out_gemm_residual_fp32(
                z_c,
                x_out,
                module.linear_g.weight,
                module.linear_z.weight,
                None,
                ln_weight=ln_in.weight,
                ln_bias=ln_in.bias,
                ln_stats=stats,
                eps=ln_in.eps,
                out=out_buf,
            )

        def out_from_dm(z_c=z_c, x_dm=x_dm, stats=stats, out_buf=out_buf):
            return gated_out_from_dm_residual_fp32(
                z_c,
                x_dm,
                module.linear_g.weight,
                module.linear_z.weight,
                None,
                ln_in.weight,
                ln_in.bias,
                stats,
                ln_out.weight,
                ln_out.bias,
                ln_out_eps=ln_out.eps,
                out=out_buf,
            )

        stages = (
            ("ln_stats", lambda z_c=z_c: ln_stats_fp32(z_c, ln_in.eps)),
            ("dual_ab", dual_ab),
            ("dual_single", dual_single),
            (
                "ln_transpose",
                lambda x_dm=x_dm: ln_transpose_fp32(
                    x_dm, ln_out.weight, ln_out.bias, ln_out.eps,
                ),
            ),
            ("out_alloc", out_alloc),
            ("out_preallocated", out_preallocated),
            ("out_from_dm", out_from_dm),
        )
        for stage, fn in stages:
            rows.append(
                {
                    "stage": stage,
                    "row_cap": row_cap,
                    "M": M,
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


def _make_case(
    n: int,
    *,
    outgoing: bool,
    dtype: torch.dtype,
    c_z: int,
    c_hidden: int,
):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    torch.manual_seed(2026 + n + int(outgoing))
    module = cls(c_z, c_hidden).cuda().eval().to(dtype=dtype)
    _randomize_observable_output(module)
    z = torch.randn(1, n, n, c_z, device="cuda", dtype=dtype) * 0.1
    mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
    return module, z, mask


def _configure_variant(kind: str, chunk_cap: int | None) -> None:
    _set_fused(kind in {"fused", "fused_inplace"})
    if chunk_cap is None:
        os.environ.pop("OPENFOLD3_TRIMUL_CHUNK_CAP", None)
    else:
        os.environ["OPENFOLD3_TRIMUL_CHUNK_CAP"] = str(chunk_cap)


def _run_variant(
    module,
    work: torch.Tensor,
    mask: torch.Tensor,
    *,
    kind: str,
    chunk_cap: int | None,
) -> torch.Tensor:
    if kind == "cueq":
        return module(
            work,
            mask=mask,
            inplace_safe=False,
            use_cueq_triangle_kernels=True,
        )
    if kind == "legacy_triton":
        # The non-fused legacy implementation always chunks its first
        # projection and cannot accept None. N gives a single sequence chunk,
        # which is its closest supported whole-calculation mode.
        legacy_chunk_size = work.shape[-3] if chunk_cap is None else chunk_cap
        return module(
            work,
            mask=mask,
            inplace_safe=True,
            use_triton_triangle_kernels=True,
            _add_with_inplace=False,
            _inplace_chunk_size=legacy_chunk_size,
        )
    if kind == "fused_inplace":
        out = fused_trimul_update(
            module,
            work,
            mask,
            with_add=True,
            out=work,
        )
        if out is None:
            raise RuntimeError("fused inplace path was ineligible")
        return out
    return module(
        work,
        mask=mask,
        inplace_safe=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
    )


def _write_results(path: Path | None, results: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[590, 737, 1264])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden", type=int, default=128)
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Diagnostic only. Default disables TF32 for full-fp32 comparison.",
    )
    parser.add_argument("--profile-n", type=int, default=None)
    parser.add_argument("--stage-split", action="store_true")
    parser.add_argument(
        "--subop-split",
        action="store_true",
        help="Time target-file kernels directly at full/chunk row counts.",
    )
    parser.add_argument(
        "--include-cueq",
        action="store_true",
        help="Also benchmark cuEq trimul; first call includes any length-keyed setup.",
    )
    parser.add_argument(
        "--include-legacy-triton",
        action="store_true",
        help="Also benchmark use_triton_triangle_kernels whole/chunked paths.",
    )
    parser.add_argument(
        "--exclude-eager",
        action="store_true",
        help="Skip the eager diagnostic baseline.",
    )
    parser.add_argument(
        "--include-fused-inplace",
        action="store_true",
        help="Benchmark production-style fused residual writeback into z.",
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

    variants = []
    if not args.exclude_eager:
        variants.append(("eager_default", "eager", None))
    variants.append(("fused", "fused", None))
    if args.include_fused_inplace:
        variants.append(("fused_inplace", "fused_inplace", None))
    if args.include_cueq:
        variants.append(("cueq", "cueq", None))
    if args.include_legacy_triton:
        variants.append(("legacy_triton", "legacy_triton", None))
    if args.chunk_cap:
        for cap in args.chunk_cap:
            variants.append((f"fused_chunk{cap}", "fused", cap))
            if args.include_fused_inplace:
                variants.append(
                    (f"fused_inplace_chunk{cap}", "fused_inplace", cap)
                )
            if args.include_legacy_triton:
                variants.append(
                    (f"legacy_triton_chunk{cap}", "legacy_triton", cap)
                )

    results = []
    header = (
        f"{'variant':<22}{'dir':>6}{'N':>7}{'median ms':>12}"
        f"{'std ms':>10}{'first ms':>11}{'peak U':>10}{'cell s':>9}"
        f"{'tf32':>7}"
    )
    print(header)
    print("-" * len(header))
    try:
        for n in args.n:
            for outgoing in (True, False):
                direction = "out" if outgoing else "in"
                module, z, mask = _make_case(
                    n,
                    outgoing=outgoing,
                    dtype=dtype,
                    c_z=args.c_z,
                    c_hidden=args.c_hidden,
                )
                u_bytes = n * n * args.c_z * z.element_size()

                for label, kind, chunk_cap in variants:
                    _configure_variant(kind, chunk_cap)

                    def prepare(z=z):
                        return z.clone()

                    def run(
                        work,
                        module=module,
                        mask=mask,
                        kind=kind,
                        chunk_cap=chunk_cap,
                    ):
                        return _run_variant(
                            module,
                            work,
                            mask,
                            kind=kind,
                            chunk_cap=chunk_cap,
                        )

                    try:
                        ms, std_ms, first_ms, peak, cell_wall_s = _bench(
                            prepare, run, reps=args.reps, warmup=args.warmup
                        )
                        row = {
                            "variant": label,
                            "direction": direction,
                            "N": n,
                            "dtype": args.dtype,
                            "c_z": args.c_z,
                            "c_hidden": args.c_hidden,
                            "tf32": torch.backends.cuda.matmul.allow_tf32,
                            "status": "ok",
                            "median_ms": ms,
                            "std_ms": std_ms,
                            "first_call_ms": first_ms,
                            "peak_transient_bytes": peak,
                            "peak_transient_mib": peak / 1024**2,
                            "peak_transient_U": peak / u_bytes,
                            "cell_wall_s": cell_wall_s,
                        }
                        print(
                            f"{label:<22}{direction:>6}{n:>7}{ms:>12.3f}"
                            f"{std_ms:>10.3f}{first_ms:>11.3f}"
                            f"{row['peak_transient_U']:>10.2f}"
                            f"{cell_wall_s:>9.1f}{str(row['tf32']):>7}"
                        )
                    except (RuntimeError, TypeError) as exc:
                        row = {
                            "variant": label,
                            "direction": direction,
                            "N": n,
                            "dtype": args.dtype,
                            "c_z": args.c_z,
                            "c_hidden": args.c_hidden,
                            "tf32": torch.backends.cuda.matmul.allow_tf32,
                            "status": "error",
                            "error": str(exc),
                        }
                        print(f"{label:<22}{direction:>6}{n:>7}  ERROR: {exc}")
                        gc.collect()
                        torch.cuda.empty_cache()
                    results.append(row)
                    _write_results(args.output_json, results)

                    if args.profile_n == n and row["status"] == "ok":
                        work = prepare()
                        _profile_once(
                            f"{label} N={n} {direction}",
                            lambda work=work: run(work),
                        )
                        del work

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
                                "c_z": args.c_z,
                                "c_hidden": args.c_hidden,
                                "tf32": torch.backends.cuda.matmul.allow_tf32,
                            }
                        )
                        results.append(row)
                        print(f"  {row['stage']:<22}{row['median_ms']:>10.3f} ms")
                if args.subop_split and outgoing:
                    row_caps = [None, *(args.chunk_cap or [64, 128, 256])]
                    subop_rows = _subop_split(
                        module,
                        z,
                        mask,
                        row_caps=row_caps,
                        reps=args.reps,
                        warmup=args.warmup,
                    )
                    print(f"\nTarget sub-ops: N={n} dir={direction}")
                    for row in subop_rows:
                        row.update(
                            {
                                "variant": "fused_subop",
                                "direction": direction,
                                "N": n,
                                "dtype": args.dtype,
                                "c_z": args.c_z,
                                "c_hidden": args.c_hidden,
                                "tf32": torch.backends.cuda.matmul.allow_tf32,
                            }
                        )
                        results.append(row)
                        cap_label = "full" if row["row_cap"] is None else row["row_cap"]
                        print(
                            f"  {row['stage']:<20} cap={str(cap_label):>4}"
                            f" M={row['M']:<8}{row['median_ms']:>9.3f} ms"
                        )
                print()

        if args.output_json is not None:
            _write_results(args.output_json, results)
            print(f"Saved {args.output_json}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    main()
