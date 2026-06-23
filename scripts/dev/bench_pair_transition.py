#!/usr/bin/env python
"""Microbenchmark OF3 SwiGLU pair transition variants.

This isolates the PairBlock transition path:

* eager_update: ``SwiGLUTransition.forward`` only, matching the default stage hook.
* eager_add: eager transition plus in-place residual add, matching full default
  block work.
* fused_update: fused Triton transition without residual.
* fused_inplace: fused Triton transition with residual in-place, matching optimized
  stage hook.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from statistics import median

import torch

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()


def _set_fused(enabled: bool) -> None:
    os.environ["OPENFOLD3_FUSED_SWIGLU_TRANSITION"] = "1" if enabled else "0"


def _bench(fn, *, reps: int, warmup: int) -> tuple[float, int]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return median(times), peak - base


def _profile_once(label: str, fn) -> None:
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            fn()
            torch.cuda.synchronize()

    print(f"\n=== CUDA profile: {label} ===")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=12,
            top_level_events_only=False,
        )
    )


def _make_variants(module, x, mask, mask_4d):
    def eager_update():
        _set_fused(False)
        return module(x, mask=mask)

    def eager_add():
        _set_fused(False)
        y = x.clone()
        y.add_(module(y, mask=mask))
        return y

    def fused_update():
        _set_fused(True)
        return module._transition(x, mask_4d)

    def fused_inplace():
        _set_fused(True)
        y = x.clone()
        return module._transition_inplace(y, mask=mask_4d, residual=y)

    return [
        ("eager_update", eager_update),
        ("eager_add", eager_add),
        ("fused_update", fused_update),
        ("fused_inplace", fused_inplace),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[590, 1264])
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--transition-n", type=int, default=4)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--profile-n",
        type=int,
        default=None,
        help="Also print one-call CUDA profiler tables at this length.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    from openfold3.core.model.layers.transition import SwiGLUTransition

    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    results = []
    header = (
        f"{'variant':<16}{'N':>7}{'ms':>10}{'peak U':>10}"
        f"{'peak MiB':>12}{'tf32':>7}"
    )
    print(header)
    print("-" * len(header))

    for n in args.n:
        torch.manual_seed(1234 + n)
        module = (
            SwiGLUTransition(c_in=args.c_z, n=args.transition_n)
            .cuda()
            .eval()
            .to(dtype=dtype)
        )
        x = torch.randn(1, n, n, args.c_z, device="cuda", dtype=dtype)
        mask = torch.ones(1, n, n, device="cuda", dtype=dtype)
        mask_4d = mask.unsqueeze(-1)
        u_bytes = n * n * args.c_z * 4

        variants = _make_variants(module, x, mask, mask_4d)
        for label, fn in variants:
            ms, peak = _bench(fn, reps=args.reps, warmup=args.warmup)
            row = {
                "variant": label,
                "N": n,
                "dtype": args.dtype,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "median_ms": ms,
                "peak_transient_bytes": peak,
                "peak_transient_mib": peak / 1024**2,
                "peak_transient_U": peak / u_bytes,
            }
            results.append(row)
            print(
                f"{label:<16}{n:>7}{ms:>10.3f}"
                f"{row['peak_transient_U']:>10.2f}"
                f"{row['peak_transient_mib']:>12.1f}"
                f"{str(row['tf32']):>7}"
            )
        print()

        if args.profile_n == n:
            for label, fn in variants:
                _profile_once(label, fn)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
