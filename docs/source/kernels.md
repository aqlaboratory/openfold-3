# Kernels

OpenFold3 supports some additional kernels to help accelerate model prediction and training. Kernel support is system specific, please review the kernel descriptions to see which kernels are compatible with your system.

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

## Triton ROCm Kernel

To use AMD ROCm-compatible Triton kernels, first install the ROCm PyTorch wheel (which bundles ROCm Triton), then install openfold3:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install openfold3
```

For AMD system installation: After installation, verify your ROCm environment is correctly configured:

```bash
validate-openfold3-rocm
```

## Smooth LDDT Ball Query Kernel

OF3 supports a CUDA ball-query backend for the smooth lDDT loss. For long targets (roughly `n_atom >= 1500`), it replaces the `O(N^2)` dense pairwise-distance scan with a sparse `[N, K]` neighbor list. The default backend remains `dense`. Enabling the kernel is opt-in.

The kernel is JIT-compiled with `ninja` on first use. Install via pixi only: the existing CUDA environments (`openfold3-cuda12`, `openfold3-cuda13`, `openfold3-cuda12-pypi`, `openfold3-cuda13-pypi`) already include the `smooth-lddt-kernel` feature (`ninja`, `nvcc`, and the required CUDA headers). No separate env names or pip extra are needed.

```bash
pixi run -e openfold3-cuda12 pytest openfold3/tests/test_diffusion_loss.py -v
```

Then enable it via the runner YAML by setting the diffusion-loss backend and a top-K:

```yaml
model_update:
  custom:
    architecture:
      loss_module:
        diffusion:
          chunk_size: null
          smooth_lddt_backend: ball_query
          smooth_lddt_top_k: 128
```

`smooth_lddt_top_k` sets the per-atom neighbor cap. `top_k = 128` or `256` is a good default for typical protein-density training samples. If `top_k` exceeds every atom's in-radius neighbor count, the result is bit-equivalent (up to floating-point reorder) to the dense backend; otherwise the kernel returns an unbiased uniform-random size-`K` subsample of the in-radius neighbors, so the loss remains an unbiased estimator of the dense value.

The ball-query backend is incompatible with the existing `chunk_size` low-memory path and will raise a clear error if both are configured.
