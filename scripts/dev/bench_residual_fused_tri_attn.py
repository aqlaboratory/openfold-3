#!/usr/bin/env python
"""Microbenchmark the residual-fused V1 triangle attention path.

Measures peak allocated bytes and wall time for the ``tri_att_start`` +
``tri_att_end`` pair at N=1264 (homo_1200-shaped), c_z=128, c_hidden=32, H=4,
chunk_size=128 across three configurations:

    v1_baseline            V1 kernel, no residual arg (allocates fresh out,
                           caller adds).  Matches the pre-patch behaviour.
    v1_residual_fused_off  V1 kernel, residual=z passed but the flag disables
                           in-place; V1 returns None and eager fallback in
                           TriangleAttention.forward folds residual.add_(x).
    v1_residual_fused_on   V1 kernel + residual=z + flag on.  ``addmm`` writes
                           per-row-block into z's storage; no ``out`` alloc.

The peak-drop target is 1U (= N² * c_z * 4 bytes ≈ 780 MiB at N=1264).
The wall-time gate is +/-5% relative to v1_baseline.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _mib(b): return b / 1024 ** 2


def _run_config(cfg: str, N: int, cap: int, warmup: int, iters: int):
    """Run one config in a subprocess (per your CLAUDE.md pattern) to isolate
    allocator state.  Returns dict of peak/wall for both starting and ending."""
    import subprocess
    env = os.environ.copy()
    env["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1"
    env["OPENFOLD3_TRI_ATTN_CHUNK_CAP"] = str(cap)
    env["OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN"] = {
        "v1_baseline": "1",  # unused because we won't pass residual
        "v1_residual_fused_off": "0",
        "v1_residual_fused_on": "1",
        "v1_case_c": "1",  # starting=True on pre-transposed z (matches tri_att_end in PairBlock)
        "cueq_baseline": "0",
        "cueq_case_c": "0",
    }[cfg]
    if cfg in ("cueq_baseline", "cueq_case_c"):
        # cuEq path: disable V1 fusion.
        env["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "0"

    script = f"""
import json
import torch
from openfold3.core.model.layers.triangular_attention import TriangleAttention

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(42)

N = {N}
CAP = {cap}
CFG = "{cfg}"
WARMUP = {warmup}
ITERS = {iters}

def make(starting):
    m = TriangleAttention(c_in=128, c_hidden=32, no_heads=4, starting=starting)
    with torch.no_grad():
        m.mha.linear_o.weight.normal_(mean=0.0, std=0.02)
        m.mha.linear_g.weight.normal_(mean=0.0, std=0.02)
    return m.cuda().eval()

out_results = {{}}

# case_c: always run with starting=True module on pre-transposed z.
# Otherwise iterate (True, False) as usual.
if CFG in ("v1_case_c", "cueq_case_c"):
    starting_iter = [True]  # module.starting = True in production tri_att_end
else:
    starting_iter = [True, False]

for starting in starting_iter:
    m = make(True) if CFG in ("v1_case_c", "cueq_case_c") else make(starting)
    x = torch.randn(1, N, N, 128, device="cuda", dtype=torch.float32)
    mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    def _step():
        z = x.clone()
        if CFG == "v1_baseline":
            with torch.no_grad():
                z_in = z if starting else z.transpose(-2, -3)
                out = m(z_in, mask=mask, chunk_size=CAP)
            return (z + out) if starting else (z + out.transpose(-2, -3) if not starting else z + out)
        elif CFG == "cueq_baseline":
            # cuEq path with chunked layer + inplace_safe (matches the eager
            # fallback in TriangleAttention.forward when V1 is off).  Caller
            # pattern: ``z = add(z, tri_att(z), inplace=True)``.  Under
            # inplace_safe, ``_chunk`` reuses z's storage via ``_out=x``.
            with torch.no_grad():
                z_in = z if starting else z.transpose(-2, -3)
                out = m(z_in, mask=mask, chunk_size=CAP,
                         use_cueq_triangle_kernels=True, inplace_safe=True)
            return z_in + out
        elif CFG == "cueq_case_c":
            # Reproduce production tri_att_end but with cuEq eager path.
            with torch.no_grad():
                z_t = z.transpose(-2, -3)
                out = m(z_t, mask=mask.transpose(-1, -2), chunk_size=CAP,
                         use_cueq_triangle_kernels=True, inplace_safe=True)
                z_t.add_(out)
                return z_t
        elif CFG == "v1_case_c":
            # Simulate production tri_att_end call: module has starting=True,
            # caller pre-transposes z, passes residual=z_transposed (non-contig).
            with torch.no_grad():
                z_t = z.transpose(-2, -3)
                return m(z_t, mask=mask.transpose(-1, -2), chunk_size=CAP, inplace_safe=True, residual=z_t)
        else:
            with torch.no_grad():
                z_in = z if starting else z.transpose(-2, -3)
                res_in = z if starting else z.transpose(-2, -3)
                return m(z_in, mask=mask, chunk_size=CAP, inplace_safe=True, residual=res_in)

    for _ in range(WARMUP):
        _step()
    torch.cuda.synchronize()

    walls = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    for _ in range(ITERS):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _step()
        e1.record()
        torch.cuda.synchronize()
        walls.append(e0.elapsed_time(e1))
    peak = torch.cuda.max_memory_allocated()
    out_results[str(starting)] = {{
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "peak_over_baseline_bytes": peak - baseline,
        "wall_ms_mean": sum(walls) / len(walls),
        "wall_ms_min": min(walls),
    }}
    del m, x, mask
    torch.cuda.empty_cache()

print("__RES__" + json.dumps(out_results))
"""
    r = subprocess.run(
        ["python", "-c", script],
        capture_output=True, env=env, cwd=str(REPO),
    )
    if r.returncode != 0:
        print(r.stderr.decode(), file=sys.stderr)
        raise RuntimeError(f"subprocess failed for {cfg}")
    import json
    for line in r.stdout.decode().splitlines():
        if line.startswith("__RES__"):
            return json.loads(line[len("__RES__"):])
    raise RuntimeError("no result marker")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1264)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=6)
    args = parser.parse_args()

    u_bytes = args.n * args.n * 128 * 4
    print(f"N={args.n}  cap={args.cap}  1U = {_mib(u_bytes):.1f} MiB")

    configs = ("cueq_baseline", "v1_baseline", "v1_residual_fused_on", "cueq_case_c", "v1_case_c")
    results = {}
    for cfg in configs:
        print(f"\n== {cfg} ==")
        results[cfg] = _run_config(cfg, args.n, args.cap, args.warmup, args.iters)
        for starting_str, d in results[cfg].items():
            print(f"  starting={starting_str}  "
                  f"peak_over_baseline={_mib(d['peak_over_baseline_bytes']):8.1f} MiB "
                  f"= {d['peak_over_baseline_bytes'] / u_bytes:5.2f}U   "
                  f"wall_min={d['wall_ms_min']:6.2f}ms  wall_mean={d['wall_ms_mean']:6.2f}ms")

    # Summary
    print("\n== Summary (starting node) ==")
    print(f"{'cfg':<28s} {'peak_delta':>15s} {'wall_mean_ms':>14s} {'vs_baseline':>14s}")
    base_key = "cueq_baseline" if "cueq_baseline" in results else configs[0]
    base = results[base_key]["True"]
    base_label = f"vs {base_key}"
    print(f"    (baseline: {base_key})")
    for cfg in configs:
        d = results[cfg].get("True")
        if d is None:
            continue
        delta_bytes = d["peak_over_baseline_bytes"] - base["peak_over_baseline_bytes"]
        delta_u = delta_bytes / u_bytes
        wall_ratio = d["wall_ms_mean"] / base["wall_ms_mean"]
        print(f"{cfg:<28s} {_mib(d['peak_over_baseline_bytes']):>10.1f} MiB "
              f"({delta_u:+.2f}U) {d['wall_ms_mean']:>10.2f}ms  {wall_ratio:>10.2%}")


if __name__ == "__main__":
    main()
