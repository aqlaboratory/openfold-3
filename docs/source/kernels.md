## CUEquivarince Kernels 

OF3 supports the CUEquivariance [triangle_multiplicative_update](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) and [triangle_attention](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html) kernels which can speed up inference/training of the model. To enable, first ensure OF3 is properly installed. Then run:

```bash
cd openfold3/
pip install .['cuequivariance'] --no-build-isolation
python setup.py install 
```

Using these kernels requires upgrading the torch and cuda version, and so a second install is requried to re-compile CUDA extensions. Then, to enable these kernels via the runnel yaml, add the following:

```yaml
model_update:
  preset: "predict"
  compile: false
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
```

This is specifically for inference, but something similar can be used for training. 