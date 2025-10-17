## CuEquivarince Kernels 

OF3 supports CuEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model.
Note: CuEquivariance acceleration can be used while DeepSpeed acceleration is enabled. 
      CuEquivariance would take precedence, and then would fall back to either DeepSpeed (if enabled) or Pytorch for the shapes it does not handle efficiently.
      Notably, it would fall back for shorter sequences (threshold controlled by CUEQ_TRIATTN_FALLBACK_THRESHOLD environment variable), and for sahpes with hidden dimenson > 128 (diffusion transformer shapes).

To enable, first ensure OF3 is properly installed. Then run:

```bash
cd openfold3/
pip install .['cuequivariance'] --no-build-isolation
python setup.py install 
```

Using these kernels requires upgrading the torch and cuda version, and so a second install is requried to re-compile CUDA extensions. Then, to enable these kernels via the runner yaml, add the following:

```yaml
model_update:
  preset: "predict"
  compile: false
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: true
```

This is specifically for inference, but something similar can be used for training. 