# cuEquivariance Kernels 

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

## Fused Triton inference kernels

OpenFold3 also provides Triton paths designed to accelerate low-memory
inference on AMD GPUs through ROCm. These paths require a Triton-enabled GPU
build of PyTorch and disabled gradient tracking. Most are disabled by default;
the table below records each default. Ineligible calls preserve the existing
implementation as a fallback.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `OPENFOLD3_FUSED_LN_LINEAR` | `0` | Fuses selected LayerNorm-to-linear projections. |
| `OPENFOLD3_FUSED_SWIGLU_TRANSITION` | `0` | Fuses the Pairformer and template pair transitions and avoids materializing their hidden expansion. |
| `OPENFOLD3_FUSED_TRI_ATTN_V1` | `0` | Enables cap-blocked triangle attention for supported Pairformer and template shapes. |
| `OPENFOLD3_TRI_ATTN_CHUNK_CAP` | unset | Caps the number of triangle-attention rows processed per call. Smaller values reduce peak memory. |
| `OPENFOLD3_FUSED_TRIMUL` | `0` | Enables the length-generic, low-memory Triton alternative to cuEquivariance triangle multiplication. |
| `OPENFOLD3_TRIMUL_CHUNK_CAP` | unset | Enables additional row/K chunking for triangle multiplication. This lowers peak memory at some runtime cost. |
| `OPENFOLD3_FUSED_DIFFUSION_ATTN` | `0` | Streams diffusion-attention K/V blocks instead of materializing the complete attention matrix. |
| `OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS` | `1024` | Sets the single-sample token threshold for fused diffusion attention. Multi-sample inference always uses it when enabled and eligible. |
| `OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE` | `0` | Reuses pair-bias projections across diffusion steps. This improves speed but adds an N²-scaled resident cache for every diffusion block. |
| `OPENFOLD3_FUSED_TEMPLATE_EMBED` | `1` | Enables eligible inference template-embedding fast paths when Triton is available. |
| `OPENFOLD3_INPLACE_OPM` | `0` | Accumulates chunked MSA outer-product-mean updates directly into the pair representation. |

The inference input embedder also uses an automatically selected fused
relative-position gather-add kernel on eligible GPU inputs. It does not have
an environment toggle.

The following example enables the main fused paths with bounded triangle
attention memory:

```bash
OPENFOLD3_FUSED_LN_LINEAR=1 \
OPENFOLD3_FUSED_SWIGLU_TRANSITION=1 \
OPENFOLD3_FUSED_TRI_ATTN_V1=1 \
OPENFOLD3_TRI_ATTN_CHUNK_CAP=128 \
OPENFOLD3_FUSED_TRIMUL=1 \
OPENFOLD3_FUSED_DIFFUSION_ATTN=1 \
OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE=0 \
OPENFOLD3_FUSED_TEMPLATE_EMBED=1 \
OPENFOLD3_INPLACE_OPM=1 \
run_openfold predict \
  --query-json=examples/example_inference_inputs/query_ubiquitin.json \
  --runner-yaml=examples/example_runner_yamls/triton.yml
```

For a lower-memory triangle-multiplication path, additionally set
`OPENFOLD3_TRIMUL_CHUNK_CAP=128`. Keep the diffusion pair-bias cache disabled
when pushing maximum sequence length: its memory usage scales with the number
of diffusion blocks and attention heads as well as N².

### Coordinate-derived template features

Template inference can avoid carrying precomputed N² distogram and unit-vector
features by enabling the compact coordinate representation in the runner
configuration:

```yaml
dataset_config_kwargs:
  template:
    use_coordinate_pair_features: true
```

This mode carries pseudo-beta and backbone-frame coordinates and constructs
their projected pair features directly on the GPU. It currently supports
inference only and requires the default template distogram settings
(`min_bin=3.25`, `max_bin=50.75`, and `n_bins=39`). The default configuration
continues to use the legacy template pair features.

### Selection and benchmarking notes

- Fused triangle attention currently supports the production trunk and
  template channel configurations. Other shapes fall back automatically.
- The fused kernels use fixed operation-signature configurations so that a
  change in target length does not normally trigger another Triton compile.
  The first use of each operation signature can still include JIT compilation
  overhead.
- Fused trimul provides explicit row/K chunking, bounded workspace, and avoids
  shape-specific tuning for each sequence length.
- `OPENFOLD3_TRI_ATTN_CHUNK_CAP` and
  `OPENFOLD3_TRIMUL_CHUNK_CAP` are memory/runtime trade-offs rather than
  universally faster settings.
- The Triton paths use PyTorch's ROCm-compatible GPU interface and are intended
  for AMD GPU inference.
- These paths are inference optimizations. Training and autograd continue to
  use the differentiable standard implementations.
