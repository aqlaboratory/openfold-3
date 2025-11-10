# OpenFold3 File Structure and Key File Paths for MLX Porting

## Entry Points

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `openfold3/run_openfold.py` | Main CLI interface | `cli()`, `train()`, `predict()`, `align_msa_server()` |
| `openfold3/setup_openfold.py` | Setup and initialization | Model download, checkpoint setup |

## Core Model Architecture

### Main Model Class
| File | Purpose | Key Classes | LOC |
|------|---------|------------|-----|
| `openfold3/projects/of3_all_atom/model.py` | Main AF3 model | `OpenFold3` | 27.7K |

### Primitives (Basic Building Blocks)
| File | Purpose | Key Classes | LOC |
|------|---------|------------|-----|
| `openfold3/core/model/primitives/attention.py` | Attention layers | `Attention`, `GlobalAttention`, `_attention()`, `_deepspeed_evo_attn()`, `_cueq_triangle_attn()` | 672 |
| `openfold3/core/model/primitives/linear.py` | Linear layers with custom init | `Linear` | 150+ |
| `openfold3/core/model/primitives/activations.py` | Activation functions | `SwiGLU` | 60+ |
| `openfold3/core/model/primitives/normalization.py` | Norm layers | `LayerNorm`, `AdaLN` | 150+ |
| `openfold3/core/model/primitives/dropout.py` | Dropout variations | `Dropout`, `RowwiseDropout`, `ColumnwiseDropout` | - |
| `openfold3/core/model/primitives/initialization.py` | Custom weight init | Init functions | - |

### Attention Variants & Updates
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/model/layers/triangular_attention.py` | Triangle attention | `TriangleAttention` |
| `openfold3/core/model/layers/triangular_multiplicative_update.py` | Triangle multiplicative update | `TriangleMultiplicativeUpdate` |
| `openfold3/core/model/layers/msa.py` | MSA attention layers | `MSARowAttention`, `MSAColumnAttention` |
| `openfold3/core/model/layers/attention_pair_bias.py` | Pair bias computation | `AttentionPairBias` |
| `openfold3/core/model/layers/outer_product_mean.py` | Outer product for MSA | `OuterProductMean` |
| `openfold3/core/model/layers/transition.py` | Transition blocks | `Transition` |
| `openfold3/core/model/layers/sequence_local_atom_attention.py` | Atom attention | `AtomAttentionEncoder`, `AtomAttentionDecoder` |

### Latent Representation Stacks
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/model/latent/msa_module.py` | MSA processing stack | `MSAModuleStack`, `MSABlock` |
| `openfold3/core/model/latent/pairformer.py` | Pair representation stack | `PairFormerStack` |
| `openfold3/core/model/latent/evoformer.py` | Evoformer block | `EvoformerBlock`, `EvoformerStack` |
| `openfold3/core/model/latent/template_module.py` | Template embeddings | `TemplateEmbedderAllAtom` |
| `openfold3/core/model/latent/base_blocks.py` | Base block classes | `MSABlock`, `PairBlock` |
| `openfold3/core/model/latent/base_stacks.py` | Base stack classes | `MSAStack`, `PairStack` |

### Diffusion & Structure
| File | Purpose | Key Classes | LOC |
|------|---------|------------|-----|
| `openfold3/core/model/structure/diffusion_module.py` | Diffusion for structure | `DiffusionModule`, `SampleDiffusion` | 12.5K |
| `openfold3/core/model/layers/diffusion_transformer.py` | Transformer for diffusion | `DiffusionTransformer` | - |
| `openfold3/core/model/layers/diffusion_conditioning.py` | Diffusion conditioning | `DiffusionConditioning` | - |

### Input Processing
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/model/feature_embedders/input_embedders.py` | Input encoding | `InputEmbedderAllAtom`, `MSAModuleEmbedder` |

### Output/Prediction Heads
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/model/heads/head_modules.py` | Confidence heads | `AuxiliaryHeadsAllAtom` |
| `openfold3/core/model/heads/prediction_heads.py` | Individual prediction heads | `pLDDTHead`, `PAEHead`, `PDEHead`, etc. |

## GPU-Accelerated Kernels (To Replace)

| File | Purpose | Kernel Type | Fallback |
|------|---------|------------|----------|
| `openfold3/core/kernels/triton/triton_softmax.py` | Softmax kernels | Triton JIT | Standard PyTorch softmax |
| `openfold3/core/kernels/triton/swiglu.py` | SwiGLU fusion | Triton JIT | Standard SiLU + mul |
| `openfold3/core/kernels/triton/fused_softmax.py` | Fused operations | Triton JIT | Unfused ops |

## Training & Inference Infrastructure

### Runners
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/runners/model_runner.py` | Base runner | `ModelRunner` |
| `openfold3/projects/of3_all_atom/runner.py` | AF3-specific runner | `OpenFold3AllAtom` |
| `openfold3/entry_points/experiment_runner.py` | Experiment orchestration | `TrainingExperimentRunner`, `InferenceExperimentRunner` |

### Loss Functions
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/core/loss/loss_module.py` | Master loss | `OpenFold3Loss` |
| `openfold3/core/loss/diffusion.py` | Diffusion loss | - |
| `openfold3/core/loss/confidence.py` | Confidence loss | - |
| `openfold3/core/loss/distogram.py` | Distance prediction loss | - |
| `openfold3/core/loss/loss_utils.py` | Loss utilities | - |

### Metrics & Validation
| File | Purpose | Key Functions |
|------|---------|-----------------|
| `openfold3/core/metrics/validation_all_atom.py` | Structure metrics | `get_metrics()`, `get_metrics_chunked()` |
| `openfold3/core/metrics/confidence.py` | Confidence metrics | - |
| `openfold3/core/metrics/rasa.py` | RASA metrics | - |
| `openfold3/core/metrics/model_selection.py` | Model selection | - |

## Utilities

### Tensor Operations
| File | Purpose | Key Functions |
|------|---------|-----------------|
| `openfold3/core/utils/tensor_utils.py` | Tensor helpers | `add()`, `tensor_tree_map()`, `permute_final_dims()`, etc. |
| `openfold3/core/utils/chunk_utils.py` | Chunking/batching | `chunk_layer()` |

### Geometry & Structure
| File | Purpose | Key Functions |
|------|---------|-----------------|
| `openfold3/core/utils/geometry/rigid_matrix_vector.py` | Rigid transforms | Quaternion/rotation operations |
| `openfold3/core/utils/geometry/kabsch_alignment.py` | Alignment | Kabsch algorithm |
| `openfold3/core/utils/rigid_utils.py` | Rigid utils | `quat_to_rot()`, etc. |

### Training Utilities
| File | Purpose | Key Functions |
|------|---------|-----------------|
| `openfold3/core/utils/checkpointing.py` | Activation checkpointing | `checkpoint_blocks()`, `get_checkpoint_fn()` |
| `openfold3/core/utils/precision_utils.py` | Mixed precision | `OF3DeepSpeedPrecision` |
| `openfold3/core/utils/lr_schedulers.py` | Learning rate scheduling | `AlphaFoldLRScheduler` |
| `openfold3/core/utils/callbacks.py` | Training callbacks | - |
| `openfold3/core/utils/exponential_moving_average.py` | EMA tracking | - |

## Configuration & Data

### Configuration
| File | Purpose | Key Classes |
|------|---------|------------|
| `openfold3/projects/of3_all_atom/config/model_config.py` | Model architecture config | Model dimension settings |
| `openfold3/projects/of3_all_atom/config/inference_query_format.py` | Inference input format | `InferenceQuerySet` |
| `openfold3/core/config/config_utils.py` | Config utilities | YAML loading/saving |
| `openfold3/core/config/msa_pipeline_configs.py` | MSA pipeline config | - |
| `openfold3/core/config/default_linear_init_config.py` | Linear init defaults | - |

### Data Processing
| File | Purpose |
|------|---------|
| `openfold3/core/data/io/sequence/` | Sequence I/O |
| `openfold3/core/data/io/structure/` | Structure I/O |
| `openfold3/core/data/pipelines/featurization/` | Feature extraction |
| `openfold3/core/data/tools/colabfold_msa_server.py` | MSA server interface |

## Entry Point Setup
| File | Purpose | Key Functions |
|------|---------|-----------------|
| `openfold3/entry_points/import_utils.py` | Import management | `_torch_gpu_setup()` |
| `openfold3/entry_points/validator.py` | Config validation | - |
| `openfold3/hacks.py` | Initialization hacks | `prep_deepspeed()`, `prep_cutlass()` |

## Project Structure
| File | Purpose |
|------|---------|
| `openfold3/projects/of3_all_atom/project_entry.py` | Project entry point |
| `openfold3/projects/of3_all_atom/constants.py` | Project constants |

## Testing
| File | Purpose |
|------|---------|
| `openfold3/tests/test_kernels.py` | Kernel correctness tests |
| `openfold3/tests/test_of3_model.py` | Full model tests |
| `openfold3/tests/test_primitives.py` | Primitive layer tests |
| `openfold3/tests/test_inference_full.py` | End-to-end inference tests |
| `openfold3/tests/compare_utils.py` | Comparison utilities |
| `openfold3/tests/data_utils.py` | Test data generation |

## Critical Patterns by Location

### Pattern 1: Kernel Selection
**Location**: `openfold3/core/model/primitives/attention.py` lines 352-372
- Selects between DeepSpeed, CuEq, LMA, or standard attention
- Fallback detection for incompatible configurations

### Pattern 2: CPU-GPU Offloading
**Location**: `openfold3/core/model/latent/evoformer.py` lines 143-149
- Moves tensors between CPU/GPU during inference
- Calls `torch.cuda.empty_cache()`

### Pattern 3: Mixed Precision Casting
**Location**: `openfold3/core/model/primitives/normalization.py` lines 59-86
- Special handling for bfloat16
- Uses `torch.amp.autocast("cuda")`

### Pattern 4: Automatic Mixed Precision
**Location**: `openfold3/core/model/latent/evoformer.py` line 121
- `torch.amp.autocast("cuda", dtype=attn_dtype)`
- Prevents precision casting issues

### Pattern 5: GPU Setup
**Location**: `openfold3/entry_points/import_utils.py`
- `torch.set_float32_matmul_precision("high")` for Ampere GPUs

## File Dependency Graph (Key Paths)

```
run_openfold.py
├── entry_points/import_utils.py (_torch_gpu_setup)
├── projects/of3_all_atom/model.py (OpenFold3)
│   ├── core/model/feature_embedders/input_embedders.py
│   ├── core/model/latent/msa_module.py
│   │   ├── core/model/layers/msa.py
│   │   ├── core/model/layers/outer_product_mean.py
│   │   └── core/model/layers/transition.py
│   ├── core/model/latent/pairformer.py
│   │   ├── core/model/layers/triangular_attention.py
│   │   │   └── core/model/primitives/attention.py
│   │   └── core/model/layers/triangular_multiplicative_update.py
│   └── core/model/structure/diffusion_module.py
└── projects/of3_all_atom/runner.py (OpenFold3AllAtom)
    └── core/loss/loss_module.py
```

## Summary Statistics

- **Total Python files**: 250
- **Total lines of code**: ~72,000
- **Files with GPU/CUDA refs**: 20
- **Major module files**: 15-20 core computation files
- **Test files**: 10+ test modules
- **Config files**: 5+ major config files

