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

# fast_trimul (optional, third-party)

OF3 can optionally route the triangle multiplicative update through
[`fast_trimul`](https://github.com/tiagomonteiro0715/fast_trimul), a fused fp16
[CuTe DSL](https://github.com/NVIDIA/cutlass) kernel. It is **inference only** (its
backward is a correct but slow torch recompute), and it holds an fp16 copy of the
layer weights, so it is opt-in and off by default.

Install the optional dependency:

```bash
pip install openfold3[fast_trimul]
```

Enable it per `TriangleMultiplicativeUpdate` module by setting the `use_fast_trimul`
attribute on the built model (before inference, after loading weights):

```python
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate,
)

model.eval()
for m in model.modules():
    if isinstance(m, TriangleMultiplicativeUpdate):
        m.use_fast_trimul = True
```

The fast path targets NVIDIA Ampere (sm80) with `N` divisible by 8; other
hardware/shapes/dtypes use fast_trimul's built-in pure-torch fallback. If the
package is not installed, or during training, the flag is ignored and the normal
OF3 path runs.
