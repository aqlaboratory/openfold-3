"""Microbenchmark AttentionPairBias.forward (the diffusion transformer's
self-attention call site) in isolation.

Compares eager vs `OPENFOLD3_FUSED_DIFFUSION_ATTN=1` at the natural
production calling convention: `a` shape [B, S, N, c_a=768], `z` shape
[B, N, N, c_z=128], with c_hidden=48, no_heads=16, use_ada_layer_norm=True.

Two regimes:
 - S=1 (smoke / single-target inference)
 - S=5 (production rollout default)

Reports per-call median ms, peak transient bytes above the resident input
tensors, and the cross-over N where the fused kernel beats eager.

Run with:
    source scripts/activate_of3.sh
    python scripts/dev/bench_diffusion_attn_module.py \
        --n 256 384 512 590 768 1024 1264 --samples 1 5 --reps 30 \
        --output-json data/inference_outputs/kernel_dev/diff_attn_baseline.json

fp32 currently OOMs the kernel SMEM budget at default block sizes; the
script skips fp32 fused runs when an OutOfResources exception fires and
records the failure in the JSON instead of crashing the whole sweep.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch


def _set_flags(*, fused: bool) -> None:
    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "1" if fused else "0"
    # Force the eligibility gate to use samples > 1 OR n_token >= cutoff;
    # we want measurements at every N regardless of the cutoff.
    os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = "0"


def _build_module(
    *, c_a: int, c_s: int, c_z: int, c_hidden: int, no_heads: int, dtype: torch.dtype
):
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias

    module = AttentionPairBias(
        c_q=c_a,
        c_k=c_a,
        c_v=c_a,
        c_s=c_s,
        c_z=c_z,
        c_hidden=c_hidden,
        no_heads=no_heads,
        use_ada_layer_norm=True,
        gating=True,
    )
    return module.cuda().eval().to(dtype=dtype)


def _bench_one(module, a, z, s, mask, *, reps: int) -> tuple[float, float]:
    """Return (median ms per call, peak transient bytes above input)."""
    with torch.no_grad():
        for _ in range(3):  # warm
            module(a=a, z=z, s=s, mask=mask)
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            module(a=a, z=z, s=s, mask=mask)
            ends[i].record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2], peak - base


def _run_one(
    *,
    label: str,
    fused: bool,
    N: int,
    S: int,
    dtype: torch.dtype,
    c_a: int,
    c_s: int,
    c_z: int,
    c_hidden: int,
    no_heads: int,
    reps: int,
) -> dict:
    _set_flags(fused=fused)

    module = _build_module(
        c_a=c_a, c_s=c_s, c_z=c_z, c_hidden=c_hidden, no_heads=no_heads, dtype=dtype,
    )

    torch.manual_seed(101 + N + 7 * S)
    B = 1
    a = torch.randn(B, S, N, c_a, device="cuda", dtype=dtype)
    s = torch.randn(B, S, N, c_s, device="cuda", dtype=dtype)
    z = torch.randn(B, N, N, c_z, device="cuda", dtype=dtype)
    mask = torch.ones(B, N, device="cuda", dtype=dtype)

    result = {
        "label": label,
        "fused": fused,
        "N": N,
        "S": S,
        "dtype": str(dtype),
        "c_a": c_a,
        "c_z": c_z,
        "c_hidden": c_hidden,
        "no_heads": no_heads,
    }

    try:
        ms, peak = _bench_one(module, a, z, s, mask, reps=reps)
        # 1U_trunk = N*N*c_z*4 bytes (fp32 reference unit)
        u_trunk = N * N * c_z * 4
        result.update(
            {
                "median_ms": ms,
                "peak_transient_bytes": peak,
                "peak_transient_U_trunk": peak / u_trunk,
                "status": "ok",
            }
        )
    except torch.cuda.OutOfMemoryError as e:
        result.update({"status": "oom", "error": str(e)[:200]})
    except Exception as e:
        # triton.runtime.errors.OutOfResources subclasses Exception, not torch.OOM.
        result.update({"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"})

    del module, a, s, z, mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[256, 384, 512, 590, 768, 1024, 1264],
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        nargs="+",
        default=["bf16"],
        help="Default bf16 only; fp32 currently OOMs the kernel SMEM budget.",
    )
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden", type=int, default=48)
    parser.add_argument("--no-heads", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dtypes = [torch.bfloat16 if d == "bf16" else torch.float32 for d in args.dtype]

    results = []
    header = (
        f"{'cfg':<10}{'dtype':>7}{'S':>3}{'N':>6}"
        f"{'ms':>10}{'peak MiB':>11}{'peak U':>9}{'status':>8}"
    )
    print(header)
    print("-" * len(header))

    for dtype, dtype_str in zip(dtypes, args.dtype):
        for S in args.samples:
            for N in args.n:
                for label, fused in (("eager", False), ("fused_v1", True)):
                    r = _run_one(
                        label=label,
                        fused=fused,
                        N=N,
                        S=S,
                        dtype=dtype,
                        c_a=args.c_a,
                        c_s=args.c_s,
                        c_z=args.c_z,
                        c_hidden=args.c_hidden,
                        no_heads=args.no_heads,
                        reps=args.reps,
                    )
                    results.append(r)
                    if r["status"] == "ok":
                        print(
                            f"{label:<10}{dtype_str:>7}{S:>3}{N:>6}"
                            f"{r['median_ms']:>10.3f}"
                            f"{r['peak_transient_bytes'] / 1024**2:>11.1f}"
                            f"{r['peak_transient_U_trunk']:>9.2f}"
                            f"{'ok':>8}"
                        )
                    else:
                        print(
                            f"{label:<10}{dtype_str:>7}{S:>3}{N:>6}"
                            f"{'-':>10}{'-':>11}{'-':>9}"
                            f"{r['status']:>8}"
                        )
            print()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
