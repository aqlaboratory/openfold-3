# cuEquivariance Kernels 

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

# Online template coordinate projection

OpenFold3 can replace resident N² template distogram / unit-vector features with compact O(N) coordinates and project pair features online in the template embedder. Enable it from dataset template settings:

```yaml
# Inference (`dataset_config_kwargs`) or training (`dataset_configs.*.*.config`)
template:
  use_coordinate_pair_features: true
```

When the batch contains `template_pseudo_beta_coords` / `template_frame_atom_coords`, the model uses the coordinate embedder automatically and streams templates on GPU (no template offload).

## Triton kernel

On CUDA with eligible shapes (`B=1`, output channels `64`, activation dtype `float32` or `bfloat16`), projection uses a length-generic Triton kernel (N and strides are not specialized). Training uses an autograd wrapper with a Triton split-K backward; inference may update the pair tensor in place. Geometry inputs stay fp32; under bf16 autocast, pair activations remain bf16 while the kernel accumulates in fp32.

```bash
# Default: use Triton when eligible
OPENFOLD3_FUSED_TEMPLATE_COORD=1

# Force the chunked eager reference path
OPENFOLD3_FUSED_TEMPLATE_COORD=0
```

TF32 for the Triton backward GEMM follows `torch.backends.cuda.matmul.allow_tf32`. Distogram settings must remain the defaults (`min_bin=3.25`, `max_bin=50.75`, `n_bins=39`).
