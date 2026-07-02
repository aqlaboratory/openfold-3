# Kernels

OpenFold3 supports some additional kernels to help accelerate model prediction and training. Kernel support is system specific, please review the kernel descriptions to see which kernels are compatible with your system.

## cuEquivariance Kernels 

OF3 supports cuEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model.
Note: cuEquivariance acceleration can be used while DeepSpeed acceleration is enabled. 
      cuEquivariance would take precedence, and then would fall back to either DeepSpeed (if enabled) or PyTorch for the shapes it does not handle efficiently.
      Notably, it would fall back for shorter sequences (threshold controlled by `CUEQ_TRIATTN_FALLBACK_THRESHOLD` environment variable), and for shapes with hidden dimension > 128 (diffusion transformer shapes).

To enable cuequivariance with pixi, use the `openfold3-cuda12-pypi` or `openfold3-cuda13-pypi` environment. Below is a example inference command

```bash
pixi run -e openfold3-cuda12-pypi run_openfold predict --query-json=query_ubiquitin.json  --runner-yaml=cuequivariance.yml
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
TODO: Add brief description and installation instructions here