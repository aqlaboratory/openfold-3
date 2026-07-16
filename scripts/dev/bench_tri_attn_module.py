"""Microbenchmark TriangleAttention.forward in isolation.

Compares V1 cap128 against cuEq / eager baselines across a small sweep of N.
Reports per-call wall time and peak transient memory above the resident input.

Run with:
    python scripts/dev/bench_tri_attn_module.py --n 384 590 --reps 20

Template-shaped (c_t=64) comparison against the current eager fallback:
    python scripts/dev/bench_tri_attn_module.py \\
      --n 384 590 1369 --c_z 64 --c_h 16 --heads 4 \\
      --skip-cueq --include-eager --include-residual --no-tf32

The dispatch flags are read at call time, so each configuration can be toggled
within one process. TF32 is enabled by default to match normal OF3 inference;
pass ``--no-tf32`` for true-fp32 diagnostics.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch


def _set_flags(*, v1: bool, cap: str | None) -> None:
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


def _call(
    module,
    x,
    mask,
    *,
    chunk_size: int | None,
    use_cueq: bool,
    residual: bool,
):
    # Residual path mutates ``x`` in place, matching PairBlock inference.
    if residual:
        return module(
            x,
            mask=mask,
            chunk_size=chunk_size,
            use_cueq_triangle_kernels=use_cueq,
            inplace_safe=True,
            residual=x,
        )
    return module(
        x,
        mask=mask,
        chunk_size=chunk_size,
        use_cueq_triangle_kernels=use_cueq,
    )


def _bench_one(
    module,
    x,
    mask,
    *,
    reps: int,
    chunk_size: int | None,
    use_cueq: bool,
    residual: bool,
) -> tuple[float, float]:
    """Return (median ms per call, peak transient bytes above input)."""
    with torch.no_grad():
        for _ in range(3):  # warm
            _call(
                module,
                x,
                mask,
                chunk_size=chunk_size,
                use_cueq=use_cueq,
                residual=residual,
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
            _call(
                module,
                x,
                mask,
                chunk_size=chunk_size,
                use_cueq=use_cueq,
                residual=residual,
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
    v1: bool,
    cap: str | None,
    use_cueq: bool,
    residual: bool,
) -> dict:
    _set_flags(v1=v1, cap=cap)

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
        ms, peak = _bench_one(
            module,
            x,
            mask,
            reps=reps,
            chunk_size=chunk_size,
            use_cueq=use_cueq,
            residual=residual,
        )
        U = N * N * c_z * x.element_size()
        return {
            "label": label,
            "N": N,
            "c_z": c_z,
            "c_h": c_h,
            "H": H,
            "dtype": str(dtype),
            "starting": starting,
            "residual": residual,
            "median_ms": ms,
            "peak_transient_bytes": peak,
            "peak_transient_U": peak / U,
        }
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
    parser.add_argument(
        "--include-eager",
        action="store_true",
        help="Include the non-cuEq eager chunked baseline.",
    )
    parser.add_argument(
        "--include-residual",
        action="store_true",
        help="Also measure residual-fused V1 (production PairBlock path).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.no_tf32:
        os.environ["OF3_BENCH_TF32"] = "0"

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # (label, v1, cap, use_cueq, residual)
    configs = [
        ("cueq_default", False, None, True, False),
        ("cueq_cap128", False, "128", True, False),
        ("v1_cap128", True, "128", False, False),
    ]
    if args.include_eager:
        configs.insert(0, ("eager_cap128", False, "128", False, False))
    if args.include_residual:
        configs.append(("v1_cap128_production", True, "128", False, True))
    if args.skip_cueq:
        configs = [c for c in configs if not c[0].startswith("cueq")]

    results = []
    print(f"{'config':<16}{'N':>6}{'ms':>10}{'peak U':>10}{'peak MiB':>12}")
    for N in args.n:
        for label, v1, cap, use_cueq, residual in configs:
            r = _run_config(
                label,
                N=N,
                c_z=args.c_z,
                c_h=args.c_h,
                H=args.heads,
                dtype=dtype,
                starting=not args.ending,
                reps=args.reps,
                v1=v1,
                cap=cap,
                use_cueq=use_cueq,
                residual=residual,
            )
            results.append(r)
            print(
                f"{label:<16}{N:>6}{r['median_ms']:>10.2f}"
                f"{r['peak_transient_U']:>10.2f}"
                f"{r['peak_transient_bytes'] / 1024**2:>12.1f}"
            )
        print()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {args.output_json}")


if __name__ == "__main__":
    main()
