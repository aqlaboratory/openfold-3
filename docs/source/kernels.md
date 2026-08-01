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

OpenFold3 includes an opt-in Triton ball-query backend for the smooth lDDT
training loss. It replaces the dense all-pairs distance tensors with at most
`K` sampled in-radius neighbors per atom. The dense implementation remains the
default.

The backend requires Triton and a CUDA device. The existing
`openfold3-cuda12`, `openfold3-cuda13`, and corresponding `*-pypi` pixi
environments provide the required runtime.

Enable it through the diffusion-loss configuration:

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

`smooth_lddt_top_k` is the maximum number of non-self neighbors sampled within
the 15 Å protein or 30 Å nucleotide radius for each query atom and defaults to
512. If it covers every in-radius neighbor, the result matches the dense loss
up to floating-point reduction order. Lower values reduce memory and runtime
but increase sampling error; higher values move the result toward dense parity.

The ball-query backend does not support the diffusion loss `chunk_size` path;
set `architecture.loss_module.diffusion.chunk_size` to `null` when enabling it.
