"""Microbenchmark TriangleAttention.forward in isolation.

Compares V3 / V2 / cap128 (cuEq with TRI_ATTN_CHUNK_CAP) / eager-cuEq
across a small sweep of N. Reports per-call wall time, peak transient
memory above the resident input, and CUDA-event totals from a torch
profiler trace so we can see where V3 spends time.

Run with:
    OPENFOLD_CACHE=/.../data/openfold_cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python scripts/dev/bench_tri_attn_module.py --n 384 590 --reps 20

The script imports openfold3 lazily after toggling env flags so each
configuration sees fresh module imports. TF32 is enabled by default to match
normal OF3 inference; pass ``--no-tf32`` for true-fp32 diagnostics.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch


def _set_flags(*, v4: bool, v3: bool, v2: bool, v1: bool, cap: str | None) -> None:
    os.environ["OPENFOLD3_FUSED_TRI_ATTN_V4"] = "1" if v4 else "0"
    os.environ["OPENFOLD3_FUSED_TRI_ATTN_V3"] = "1" if v3 else "0"
    os.environ["OPENFOLD3_FUSED_TRI_ATTN_V2"] = "1" if v2 else "0"
    os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1" if v1 else "0"
    if cap is None or cap == "":
        os.environ.pop("OPENFOLD3_TRI_ATTN_CHUNK_CAP", None)
    else:
        os.environ["OPENFOLD3_TRI_ATTN_CHUNK_CAP"] = cap


def _build_module(c_in: int, c_hidden: int, no_heads: int, starting: bool):
    from openfold3.core.model.layers.triangular_attention import TriangleAttention

    module = TriangleAttention(
        c_in=c_in, c_hidden=c_hidden, no_heads=no_heads, starting=starting,
    )
    with torch.no_grad():
        torch.manual_seed(1234 + int(starting))
        module.mha.linear_o.weight.normal_(mean=0.0, std=0.02)
        if module.mha.linear_g is not None:
            module.mha.linear_g.weight.normal_(mean=0.0, std=0.02)
    return module.cuda().eval()


def _bench_one(
    module,
    x,
    mask,
    *,
    reps: int,
    chunk_size: int | None,
) -> tuple[float, float]:
    """Return (median ms per call, peak transient bytes above input)."""
    with torch.no_grad():
        for _ in range(3):  # warm
            module(
                x,
                mask=mask,
                chunk_size=chunk_size,
                use_cueq_triangle_kernels=True,
            )
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        # Use CUDA events for accurate per-call timing.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            module(
                x,
                mask=mask,
                chunk_size=chunk_size,
                use_cueq_triangle_kernels=True,
            )
            ends[i].record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2], peak - base


def _run_config(
    label: str,
    *,
    N: int,
    c_z: int,
    c_h: int,
    H: int,
    dtype: torch.dtype,
    starting: bool,
    reps: int,
    v4: bool,
    v3: bool,
    v2: bool,
    v1: bool,
    cap: str | None,
) -> dict:
    _set_flags(v4=v4, v3=v3, v2=v2, v1=v1, cap=cap)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(
        os.environ.get("OF3_BENCH_TF32", "1") in ("1", "true")
    )
    try:
        module = _build_module(c_in=c_z, c_hidden=c_h, no_heads=H, starting=starting)
        if dtype != torch.float32:
            module = module.to(dtype)

        torch.manual_seed(101 + N)
        x = torch.randn(1, N, N, c_z, device="cuda", dtype=dtype)
        mask = torch.ones(1, N, N, device="cuda", dtype=dtype)

        chunk_size = int(cap) if cap not in (None, "") else None
        ms, peak = _bench_one(module, x, mask, reps=reps, chunk_size=chunk_size)
        U = N * N * c_z * x.element_size()
        return {
            "label": label,
            "N": N,
            "c_z": c_z,
            "c_h": c_h,
            "H": H,
            "dtype": str(dtype),
            "starting": starting,
            "median_ms": ms,
            "peak_transient_bytes": peak,
            "peak_transient_U": peak / U,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _profile_v3_sub_ops(N: int, c_z: int, c_h: int, H: int) -> None:
    """torch.profiler trace of V3 call — dumps top kernels by CUDA time."""
    from torch.profiler import ProfilerActivity, profile

    _set_flags(v4=False, v3=True, v2=False, v1=False, cap=None)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(
        os.environ.get("OF3_BENCH_TF32", "1") in ("1", "true")
    )
    try:
        module = _build_module(c_in=c_z, c_hidden=c_h, no_heads=H, starting=True)
        x = torch.randn(1, N, N, c_z, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            for _ in range(3):
                module(x, mask=mask)
            torch.cuda.synchronize()

            with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as p:
                for _ in range(5):
                    module(x, mask=mask)
                torch.cuda.synchronize()

        print(f"\n=== V3 CUDA profile @ N={N} (5 calls) ===")
        print(
            p.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=15,
                top_level_events_only=False,
            )
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[256, 384, 590])
    parser.add_argument("--c_z", type=int, default=128)
    parser.add_argument("--c_h", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 fp32 matmuls. Default matches normal OF3 inference.",
    )
    parser.add_argument("--ending", action="store_true")
    parser.add_argument(
        "--skip-cueq",
        action="store_true",
        help="Skip the cuEq baseline (useful at large N where it OOMs).",
    )
    parser.add_argument("--profile-n", type=int, default=None,
                        help="If set, also run a torch.profiler trace at this N.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.no_tf32:
        os.environ["OF3_BENCH_TF32"] = "0"

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    configs = [
        # (label, v4, v3, v2, cap)
        ("cueq_default",       False, False, False, False, None),
        ("cueq_cap128",        False, False, False, False, "128"),
        ("v1_cap128",          False, False, False, True,  "128"),
        ("v2",              False, False, True,  False, None),
        ("v3",              False, True,  False, False, None),
        ("v4",              True,  False, False, False, None),
    ]
    if args.skip_cueq:
        configs = [c for c in configs if not c[0].startswith("cueq")]

    results = []
    print(f"{'config':<14}{'N':>6}{'ms':>10}{'peak U':>10}{'peak MiB':>12}")
    for N in args.n:
        for label, v4, v3, v2, v1, cap in configs:
            r = _run_config(
                label,
                N=N, c_z=args.c_z, c_h=args.c_h, H=args.heads,
                dtype=dtype, starting=not args.ending, reps=args.reps,
                v4=v4, v3=v3, v2=v2, cap=cap,
                v1=v1,
            )
            results.append(r)
            print(
                f"{label:<14}{N:>6}{r['median_ms']:>10.2f}"
                f"{r['peak_transient_U']:>10.2f}"
                f"{r['peak_transient_bytes'] / 1024**2:>12.1f}"
            )
        print()

    if args.profile_n is not None:
        _profile_v3_sub_ops(args.profile_n, args.c_z, args.c_h, args.heads)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {args.output_json}")


if __name__ == "__main__":
    main()
