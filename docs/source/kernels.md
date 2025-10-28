# cuEquivariance Kernels 

OF3 supports cuEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model.
Note: cuEquivariance acceleration can be used while DeepSpeed acceleration is enabled. 
      cuEquivariance would take precedence, and then would fall back to either DeepSpeed (if enabled) or PyTorch for the shapes it does not handle efficiently.
      Notably, it would fall back for shorter sequences (threshold controlled by `CUEQ_TRIATTN_FALLBACK_THRESHOLD` environment variable), and for shapes with hidden dimension > 128 (diffusion transformer shapes).

To enable, first install OpenFold3 with cuEquivariance: 

```bash
pip install openfold3[cuequivariance]
```

Using these kernels requires upgrading the torch and cuda versions, and so a second install is required to re-compile CUDA extensions. Then, to enable these kernels via the runner yaml, add the following:

```yaml
model_update:
  preset: 
    - "predict"
    - "pae_enabled"  # if using PAE enabled model
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: true
```

This is specifically for inference, but similar settings can be used for training. 