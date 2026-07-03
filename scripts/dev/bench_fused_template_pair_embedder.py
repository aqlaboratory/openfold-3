#!/usr/bin/env python
"""Microbenchmark for the fused template pair embedder Triton kernel.

Reports per-call wall time and peak transient bytes above input tensors, for
the eager (``TemplatePairEmbedderAllAtom.forward`` single-template) path and
the fused ``fused_template_pair_embedder_inference`` path, across a sweep of
sequence lengths ``N``.

Usage:
    python scripts/dev/bench_fused_template_pair_embedder.py \
        --n 384 590 1264 --reps 20 --dtype fp32 \
        --output-json data/inference_outputs/profiling/bench_fused_template_pair_embedder.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")

import torch

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _mib(b):
    return b / 1024**2


def _u_bytes(N: int, c_z: int = 128, dtype=torch.float32) -> int:
    return N * N * c_z * torch.tensor([], dtype=dtype).element_size()


def _make_module(dtype: torch.dtype):
    from openfold3.core.model.feature_embedders.template_embedders import (
        TemplatePairEmbedderAllAtom,
    )
    torch.manual_seed(0)
    m = TemplatePairEmbedderAllAtom(
        c_in=128, c_dgram=39, c_aatype=32, c_out=64
    ).cuda().eval()
    if dtype != torch.float32:
        m = m.to(dtype)
    return m


def _make_batch(N: int, dtype: torch.dtype, seed: int = 0) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = 1
    batch = {
        "template_distogram": torch.randn(B, 1, N, N, 39, generator=g, device="cuda", dtype=dtype),
        "template_restype": torch.randn(B, 1, N, 32, generator=g, device="cuda", dtype=dtype),
        "template_pseudo_beta_mask": torch.rand(B, 1, N, generator=g, device="cuda", dtype=dtype),
        "template_backbone_frame_mask": torch.rand(B, 1, N, generator=g, device="cuda", dtype=dtype),
        "template_unit_vector": torch.randn(B, 1, N, N, 3, generator=g, device="cuda", dtype=dtype),
        "asym_id": torch.zeros(B, N, device="cuda", dtype=torch.long),
    }
    return batch


def _bench_once(fn, *args, warm: int = 3, reps: int = 20, **kwargs):
    """Return (median_ms, peak_transient_bytes)."""
    for _ in range(warm):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()
    # Memory
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    transient = peak - baseline
    del out
    torch.cuda.empty_cache()

    # Timing
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(reps)
    ]
    torch.cuda.synchronize()
    for start, end in events:
        start.record()
        _ = fn(*args, **kwargs)
        end.record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in events)
    median = times_ms[len(times_ms) // 2]
    return median, transient


def bench_N(N: int, dtype: torch.dtype, reps: int) -> dict:
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    module = _make_module(dtype)
    batch = _make_batch(N=N, dtype=dtype)
    z = torch.randn(1, N, N, 128, device="cuda", dtype=dtype)

    with torch.inference_mode():
        # Force eager path in module.forward for the baseline (otherwise the
        # module dispatches internally to the fused wrapper).
        os.environ["OPENFOLD3_FUSED_TEMPLATE_EMBED"] = "0"
        eager_ms, eager_bytes = _bench_once(
            module, batch, z, warm=3, reps=reps
        )
        os.environ["OPENFOLD3_FUSED_TEMPLATE_EMBED"] = "1"
        fused_ms, fused_bytes = _bench_once(
            fused_template_pair_embedder_inference,
            warm=3, reps=reps,
            module=module, batch=batch, z=z, template_index=0,
        )

    U = _u_bytes(N, c_z=128, dtype=dtype)
    row = {
        "N": N,
        "dtype": str(dtype),
        "1U_MiB": round(_mib(U), 3),
        "eager_ms": round(eager_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "eager_transient_MiB": round(_mib(eager_bytes), 3),
        "fused_transient_MiB": round(_mib(fused_bytes), 3),
        "eager_transient_U": round(eager_bytes / U, 3),
        "fused_transient_U": round(fused_bytes / U, 3),
        "speedup": round(eager_ms / fused_ms, 3) if fused_ms > 0 else float("nan"),
        "memory_savings_MiB": round(_mib(eager_bytes - fused_bytes), 3),
        "memory_savings_U": round((eager_bytes - fused_bytes) / U, 3),
    }
    del module, batch, z
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[384, 590, 737, 1264])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp32")
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]

    print(f"{'N':>6s} {'eager ms':>10s} {'fused ms':>10s} {'speedup':>8s} "
          f"{'eager U':>8s} {'fused U':>8s} {'saved U':>8s} {'saved MiB':>10s}")
    print("-" * 82)
    rows = []
    for N in args.n:
        r = bench_N(N=N, dtype=dtype, reps=args.reps)
        rows.append(r)
        print(
            f"{r['N']:>6d} {r['eager_ms']:>10.3f} {r['fused_ms']:>10.3f} "
            f"{r['speedup']:>8.2f} {r['eager_transient_U']:>8.3f} "
            f"{r['fused_transient_U']:>8.3f} {r['memory_savings_U']:>8.3f} "
            f"{r['memory_savings_MiB']:>10.2f}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
