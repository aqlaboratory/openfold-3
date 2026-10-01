# Kernels

## cuEquivariance Kernels

OF3 supports cuEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model.
Note: cuEquivariance acceleration can be used while DeepSpeed acceleration is enabled. 
      cuEquivariance would take precedence, and then would fall back to either DeepSpeed (if enabled) or PyTorch for the shapes it does not handle efficiently.
      Notably, it would fall back for shorter sequences (threshold controlled by `CUEQ_TRIATTN_FALLBACK_THRESHOLD` environment variable), and for shapes with hidden dimension > 128 (diffusion transformer shapes).

To enable cuequivariance with pixi, use the `openfold3-cuda12-pypi` or `openfold3-cuda13-pypi` environment. Below is a example inference command

```bash
pixi run -e openfold3-cuda12-pypi \
  run_openfold predict --query-json=query_ubiquitin.json  --runner-yaml=cuequivariance.yml
```

For other workflows, cuequivariance must first be installed with the cuequivariance optional dependency, e.g.

```bash
pip install openfold3[cuequivariance]
```

Then, to enable these kernels via the runner.yaml, add the following:

```yaml
# cuequivariance.yml
model_update:
  presets: 
    - "predict"
    - "low_mem"  # for lower memory systems
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: true  # set this to False to use cueq only
```

This runner.yml is specifically for inference, but similar settings can be used for training.

## Smooth lDDT Ball-Query Kernel

OpenFold3 includes a Triton ball-query backend for the smooth lDDT training
loss. It replaces dense all-pairs distance tensors with sampled in-radius
neighbors and is the all-atom model default. Dense remains available as the
exact reference and chunked-execution fallback.

The backend requires Triton and a PyTorch-visible GPU. The existing
`openfold3-cuda12`, `openfold3-cuda13`, corresponding `*-pypi`, and
`openfold3-rocm7` pixi environments provide the required dependencies. ROCm
support must be verified by running the focused loss tests on AMD hardware.

Configure it through the diffusion loss:

```yaml
model_update:
  custom:
    architecture:
      loss_module:
        diffusion:
          chunk_size: null
          smooth_lddt_backend: ball_query
          smooth_lddt_top_k: 512
```

`smooth_lddt_top_k` defaults to 512. `ball_query` retains up to 512 in-radius
neighbors per atom: 15 Å for non-nucleotide centers and 30 Å for nucleotide
centers. Exact rows retain unit weight. For truncated rows, the base pair
weights use cap-dependent scaling with a lower bound of 1:

```text
protein weight    = max(1, 512 / K)
nucleotide weight = max(1, 2048 / K)
```

For `K = 256, 512, 1024, 2048`, the protein/nucleotide weights are respectively
`2/8`, `1/4`, `1/2`, and `1/1`. The same pair weight is applied to the
score numerator and pair-count denominator, keeping the loss normalized.

The legacy and current ball-query paths use the same `A * K` reservoir capacity.
The current reduction keeps only `O(A)` additional row metadata for reweighting,
so both have the same dominant `O(A * K)` storage at the same `K`.

Ball-query does not support the diffusion loss `chunk_size` path directly. If
`chunk_size` is non-null, the diffusion loss automatically uses the chunked
dense smooth lDDT calculation. Set `chunk_size: null` to run ball-query.
