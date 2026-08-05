# Smooth lDDT CUDA ball-query kernel

CUDA-only ball-query backend for the smooth lDDT loss. The dense `[N, N]`
implementation in `openfold3/core/loss/diffusion.py::smooth_lddt_loss` remains
the reference. This kernel is opt-in via
`architecture.loss_module.diffusion.smooth_lddt_backend: ball_query` and
becomes the better choice once `n_atom` is large enough that the dense `[N, N]`
matrix dominates memory.

## When to use it

Measured on a single RTX 4090 (fwd+bwd, `n_sample=2`):

| n_atom | dense fwd+bwd peak | bq top_k=256 peak | dense time | bq256 time |
|-------:|-------------------:|------------------:|-----------:|-----------:|
| 448    | 23.4 MB            | 16.4 MB           | 1.6 ms     | 5.4 ms     |
| 635    | 46.9 MB            | 23.2 MB           | 1.2 ms     | 5.4 ms     |
| 1262   | 184.9 MB           | 46.1 MB           | 2.0 ms     | 3.9 ms     |
| 3329   | 1286.0 MB          | 121.7 MB          | 16.0 ms    | 5.4 ms     |

Rule of thumb: dense for `n_atom ≲ 1000`, ball-query (`top_k ≈ 256`) for
`n_atom ≳ 2000`. Memory savings scale as `n_atom² / (n_atom · top_k)` and the
speed crossover is around `n_atom ≈ 1500`. Reproduce with
`pytest --benchmark-only scripts/benchmarks/ball-query-lddt-benchmark/test_smooth_lddt_benchmark.py`.

## Kernel layout (`ball_query_ext/`)

| Symbol | What it does |
|---|---|
| `BallQueryKernel` | Sequential per-thread ball query (legacy, kept for parity tests) |
| `BallQueryKernelCoop<W=8>` | Warp-cooperative ball query with reservoir sampling — `W` lanes scan candidates per query atom |
| `BallQueryKernelCoopWithPred<scalar_pred, W=8>` | Coop kernel that **also** writes per-pair predicted squared distances `‖pred_i − pred_j‖²` directly. Eliminates the Python-side `x_j` gather and the `(x_i − x_j)` materialization. |
| `BallQueryPredBackwardKernel<scalar_pred>` | Dedicated backward: scatters `2 · grad · (pred_i − pred_j)` into `±grad_pred` via `atomicAdd` (fp32 accumulator). Reloads positions from saved `pred` + `idx` — never materializes any `[B, N, K, 3]` tensor. |

GT positions (`p1`, `p2`) are processed in fp32 for neighbor finding.
Predictions (`pred`) are loaded in their native dtype (bf16 or fp32),
distances are computed in fp32 internally, and `dists_pred` is stored in
`pred`'s dtype — bf16 in mixed-precision training keeps the same memory
footprint as the original gather path.

The backward kernel uses `atomicAdd` on a fp32 grad buffer, which is
nondeterministic (reduction order varies). This matches PyTorch3D's
`knn_points_backward` behavior. The Python wrapper casts the fp32 grad back
to `pred.dtype` (so bf16 training sees a bf16 gradient).

## Python layer (`__init__.py`)

```
flat_x [B, N, 3]                         (requires_grad, native dtype)
   │
   ▼ _BallQueryWithPredDist.apply(...)   (custom autograd.Function)
   │     forward → CUDA kernel:  (idx, dists_gt fp32, dists_pred native)
   │     ctx saves only (pred, idx) — no x_j, no (x_i − x_j)
   │
   ▼ torch.utils.checkpoint.checkpoint(_score_from_dists, ...)
   │     pure scoring math: sqrt → abs → sigmoid → mask → reduce
   │     no autograd saves (recomputed on backward)
   ▼
loss [B, n_sample]
```

The checkpoint boundary frees every elementwise scoring intermediate (the four
sigmoid outputs, `c`, `c·e`, the masked sums). Backward recomputes scoring
from the saved `(pred, idx)` plus the masks (rebuilt cheaply from `idx`),
producing `grad_dists_pred`, which then enters the CUDA backward kernel.

`run_low_mem_loss_fn` (external chunking over the sample dim in
`diffusion.py`) still raises for the ball-query backend — chunking is
orthogonal to these internal optimizations and the restriction is unchanged.

## Files

| Path | Role |
|------|------|
| `ball_query_ext/ball_query.cu` | All CUDA kernels + launchers. BSD-3-Clause attribution preserved next to the Apache-2.0 header for the portions adapted from PyTorch3D (`_ball_query` + `knn_points_backward` pattern). |
| `ball_query_ext/ball_query.h` | C++ declarations + thin `is_cuda` checks. |
| `ball_query_ext/binding.cpp` | pybind11 bindings (`ball_query`, `ball_query_coop`, `ball_query_coop_with_pred`, `ball_query_pred_backward`). |
| `ball_query.py` | JIT loader (`torch.utils.cpp_extension.load`) + Python wrappers. |
| `__init__.py` | `_BallQueryWithPredDist` autograd Function, `_score_from_dists` checkpointed scoring, `ball_query_smooth_lddt_loss` entry point. |

## Build / debug

JIT-compiled on first use via `torch.utils.cpp_extension.load`. The kernel is
opt-in and not part of the default install — `ninja` and a matching CUDA
toolchain are required only for this backend. Prefer a pixi CUDA environment
(`openfold3-cuda12` or `openfold3-cuda13`), which already ships these
dependencies. Set `OPENFOLD3_SMOOTH_LDDT_VERBOSE=1` to get the full nvcc
build log on first call. The compute capability is read from
`torch.cuda.get_device_capability()` and exported as `TORCH_CUDA_ARCH_LIST`
only when the user has not set it explicitly.

## Tests

- Correctness + gradient parity vs dense: `pytest openfold3/tests/test_diffusion_loss.py -k "ball_query" -v`
- Benchmark sweep across real protein samples / dtypes / `top_k`:
  `pytest --benchmark-only scripts/benchmarks/ball-query-lddt-benchmark/test_smooth_lddt_benchmark.py`

The gradient parity test uses a slightly looser tolerance (`atol=1e-3`,
`rtol=5e-3`) because of the `atomicAdd` reordering and the different reduction
order between the dense `[N, N]` path and the `[N, K+1]` ball-query path.

## Attribution

The forward ball-query and the `atomicAdd`-scatter backward pattern are
modified from PyTorch3D (Meta Platforms, Inc., BSD-3-Clause) by
Liang Hong <lhong22@cse.cuhk.edu.hk>. The upstream reference is
[facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) —
specifically
[`pytorch3d/csrc/ball_query/ball_query.cu`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.cu)
and
[`pytorch3d/csrc/knn/knn.cu`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/knn/knn.cu)
under
[`LICENSE`](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE).
The BSD-3-Clause notice appears in `ball_query_ext/ball_query.cu` next to the
Apache-2.0 header for OpenFold3, and the per-file headers in this directory
call out which portions are upstream vs. OpenFold3-specific additions.
