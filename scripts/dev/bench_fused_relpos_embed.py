#!/usr/bin/env python
"""Microbenchmark for fused_relpos_embed_add_ kernel.

Compares the fused Triton kernel against the eager PyTorch path at multiple
sequence lengths, reporting correctness, peak memory, and wall time.
"""

from __future__ import annotations

import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.kernels.triton.fused_relpos_embed import (  # noqa: E402
    fused_relpos_embed_add_,
)

C = 128
VOCAB = 130
SAME_ENTITY_OFFSET = 65
WARMUP = 3
REPS = 10


def eager_relpos_add_(z, w, idx1, idx2, idx3, same_entity):
    z.add_(w[idx1])
    z.add_(w[idx2])
    z.add_(w[idx3])
    z.add_(same_entity[..., None].to(dtype=z.dtype) * w[SAME_ENTITY_OFFSET])


def bench_one(N: int) -> dict:
    torch.cuda.empty_cache()
    z = torch.randn(1, N, N, C, device="cuda", dtype=torch.float32)
    w = torch.randn(VOCAB, C, device="cuda", dtype=torch.float32)
    idx1 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx2 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    idx3 = torch.randint(0, VOCAB, (1, N, N), device="cuda", dtype=torch.int64)
    same_entity = torch.randint(0, 2, (1, N, N), device="cuda", dtype=torch.bool)

    U_bytes = N * N * C * 4

    # Correctness (skip at large N to avoid OOM from holding both z_ref and z_test)
    max_err = None
    if N <= 3000:
        z_ref = z.clone()
        eager_relpos_add_(z_ref, w, idx1, idx2, idx3, same_entity)

        z_test = z.clone()
        fused_relpos_embed_add_(z_test, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET)
        torch.cuda.synchronize()
        max_err = (z_test - z_ref).abs().max().item()
        del z_ref, z_test

    # Memory: eager
    eager_peak_U = None
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        z_tmp = z.clone()
        eager_relpos_add_(z_tmp, w, idx1, idx2, idx3, same_entity)
        torch.cuda.synchronize()
        eager_peak_U = (torch.cuda.max_memory_allocated() - base) / U_bytes
        del z_tmp
    except torch.cuda.OutOfMemoryError:
        eager_peak_U = None
        torch.cuda.empty_cache()

    # Memory: fused
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    z_tmp = z.clone()
    fused_relpos_embed_add_(z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET)
    torch.cuda.synchronize()
    fused_peak_U = (torch.cuda.max_memory_allocated() - base) / U_bytes
    del z_tmp

    # Speed: eager
    eager_ms = None
    try:
        for _ in range(WARMUP):
            z_tmp = z.clone()
            eager_relpos_add_(z_tmp, w, idx1, idx2, idx3, same_entity)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            z_tmp = z.clone()
            eager_relpos_add_(z_tmp, w, idx1, idx2, idx3, same_entity)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / REPS * 1000
        del z_tmp
    except torch.cuda.OutOfMemoryError:
        eager_ms = None
        torch.cuda.empty_cache()

    # Speed: fused
    for _ in range(WARMUP):
        z_tmp = z.clone()
        fused_relpos_embed_add_(z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        z_tmp = z.clone()
        fused_relpos_embed_add_(z_tmp, w, idx1, idx2, idx3, same_entity, SAME_ENTITY_OFFSET)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / REPS * 1000

    del z, w, idx1, idx2, idx3, same_entity
    torch.cuda.empty_cache()

    return {
        "N": N,
        "max_err": max_err,
        "eager_peak_U": eager_peak_U,
        "fused_peak_U": fused_peak_U,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
    }


def main():
    lengths = [256, 512, 1264, 2000, 3000, 3950]
    print(f"{'N':>5} | {'err':>9} | {'eager_peak':>11} | {'fused_peak':>11} | {'eager_ms':>9} | {'fused_ms':>9} | {'speedup':>7}")
    print("-" * 80)
    for N in lengths:
        r = bench_one(N)
        err_str = f"{r['max_err']:.2e}" if r["max_err"] is not None else "skip"
        eager_peak_str = f"{r['eager_peak_U']:.2f}U" if r["eager_peak_U"] is not None else "OOM"
        eager_ms_str = f"{r['eager_ms']:.1f}ms" if r["eager_ms"] is not None else "OOM"
        speedup_str = f"{r['eager_ms'] / r['fused_ms']:.2f}x" if r["eager_ms"] is not None else "N/A"
        print(
            f"{r['N']:>5} | {err_str:>9} | {eager_peak_str:>11} | {r['fused_peak_U']:.2f}U{'':>5} | {eager_ms_str:>9} | {r['fused_ms']:.1f}ms{'':>4} | {speedup_str:>7}"
        )


if __name__ == "__main__":
    main()
