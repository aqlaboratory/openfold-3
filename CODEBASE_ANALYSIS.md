# OpenFold3 Codebase Comprehensive Analysis for MLX Porting

## Executive Summary

OpenFold3 is a ~72K LOC PyTorch-based biomolecular structure prediction model targeting GPU acceleration through CUDA, DeepSpeed, Triton, and cuEquivariance kernels. The codebase is well-organized into modular components that will require systematic porting to MLX for Apple Silicon support.

---

## 1. OVERALL PROJECT STRUCTURE

### Root Organization
```
openfold-3-mlx/
├── openfold3/                    # Main package (250 Python files, ~72K LOC)
│   ├── core/                     # Core computational modules
│   ├── projects/                 # Project-specific implementations
│   ├── entry_points/             # CLI and runner interfaces
│   ├── tests/                    # Test suite
│   └── scripts/                  # Helper scripts
├── docs/                         # Documentation
├── examples/                     # Example configs and inputs
├── deepspeed_configs/            # DeepSpeed configurations
├── Dockerfile & Dockerfile.blackwell  # CUDA-based containers
└── pyproject.toml & setup.py     # Package configuration
```

### Key Entry Points
- **`run_openfold.py`**: Main CLI with three commands:
  - `train`: Training with dataset cache
  - `predict`: Inference on queries
  - `align-msa-server`: MSA alignment only

---

## 2. GPU/CUDA DEPENDENCIES AND USAGE

### Direct CUDA Dependencies
Files with explicit CUDA/GPU references: **20 core files**

#### Hardware Acceleration Libraries
1. **DeepSpeed (DS4S - DeepSpeed4Science)**
   - Location: `openfold3/core/model/primitives/attention.py` (lines 38-47)
   - Usage: `DS4Sci_EvoformerAttention` for memory-efficient attention
   - Config: `use_deepspeed_evo_attention` flag
   - Requires sequence length > 16

2. **cuEquivariance (CuEq)**
   - Location: `openfold3/core/model/primitives/attention.py` (lines 49-68)
   - Usage: `triangle_attention` kernel for equivariant computations
   - Has fallback detection for incompatible shapes
   - Incompatibility thresholds: seq_len ≤ 128 or hidden_dim constraints

3. **Triton Kernels**
   - Location: `openfold3/core/kernels/triton/`
   - Files:
     - `triton_softmax.py`: Custom softmax with mask/bias support
     - `swiglu.py`: SwiGLU activation (LigerSiLUMulFunction)
     - `fused_softmax.py`: Additional softmax variants
   - Conditional import with fallback to standard PyTorch

### GPU Memory Management
- **File**: `openfold3/core/model/latent/evoformer.py` (lines 143-149)
  - CPU-GPU offloading for inference (`_offload_inference` flag)
  - `torch.cuda.empty_cache()` calls for memory cleanup
  - Module offloading pattern: Move pair representation to CPU during intermediate steps

### Device Placement
- **File**: `openfold3/entry_points/import_utils.py`
  - `_torch_gpu_setup()`: Sets float32 matmul precision for Ampere GPUs
  - Called before training/inference starts

### Precision and Data Types
- **File**: `openfold3/core/utils/precision_utils.py`
  - `OF3DeepSpeedPrecision` class for selective input casting
  - Preserves float32 for ground truth and reference conformer features
  - Avoids truncation of atom coordinates

### Autocast Patterns
- **Locations**: 
  - `openfold3/core/model/latent/evoformer.py`: `torch.amp.autocast("cuda", ...)`
  - `openfold3/core/model/primitives/normalization.py`: Special handling for bfloat16
- Mixed precision training with DeepSpeed integration

---

## 3. KEY COMPUTATIONAL MODULES

### Architecture Layers (~15 files)

#### Attention Mechanisms (672 LOC)
**File**: `openfold3/core/model/primitives/attention.py`
- **Standard Multi-Head Attention (MHA)** class
  - Q, K, V projections with custom initializations
  - Supports 2 alternative kernels (DeepSpeed + cuEquivariance)
  - Low-Memory Attention (LMA) option for memory-constrained settings
  - Gating mechanism on output

- **Key Functions**:
  - `_attention()`: Core computation with optional high precision
  - `softmax_no_cast()`: Prevents automatic fp32 casting for bfloat16
  - `attention_chunked_trainable()`: Checkpointed attention for training
  - `_deepspeed_evo_attn()`: DeepSpeed kernel wrapper
  - `_cueq_triangle_attn()`: cuEquivariance kernel wrapper
  - `_lma()`: Low-memory chunked implementation

#### Triangle Attention (TriangleAttention)
**File**: `openfold3/core/model/layers/triangular_attention.py`
- Pair representation updates with attention bias
- Supports kernel acceleration flags
- Chunking for sequence-level processing

#### MSA Processing
**File**: `openfold3/core/model/layers/msa.py`
- MSARowAttention: Row-wise attention on MSA representations
- MSAColumnAttention: Column-wise attention
- Supports DeepSpeed EvoformerAttention kernel

#### Core Layer Types
1. **OuterProductMean** (`outer_product_mean.py`)
   - Efficient pair representation generation from MSA

2. **TriangleMultiplicativeUpdate** (`triangular_multiplicative_update.py`)
   - Pair representation updates with element-wise multiplications
   - Supports chunking and DeepSpeed kernels

3. **DiffusionTransformer** (`diffusion_transformer.py`)
   - Transformer for structure diffusion
   - 4 blocks for attention in diffusion loop

4. **Sequence Local Atom Attention** (`sequence_local_atom_attention.py`)
   - Encoder/Decoder for atom-level attention
   - Important for all-atom structure prediction

#### Activation Functions
**File**: `openfold3/core/model/primitives/activations.py`
- **SwiGLU**: 
  - Uses `LigerSiLUMulFunction` if Triton installed and on CUDA
  - Fallback: standard `nn.SiLU()` + multiplication

#### Normalization
**File**: `openfold3/core/model/primitives/normalization.py`
- **LayerNorm**: Custom implementation with DeepSpeed integration
  - Special handling for bfloat16 to avoid casting issues
  - Uses `torch.amp.autocast("cuda", enabled=False)` context
- **AdaLN** (Adaptive LayerNorm): Conditioning on auxiliary features

#### Linear Layers
**File**: `openfold3/core/model/primitives/linear.py`
- Custom initializations: lecun_normal, he_normal, glorot_uniform, gating, final
- Extends `nn.Linear` with AF3 initialization schemes

### Model Architecture Stacks

#### Latent Representations (Module Stack)
**Files**: `openfold3/core/model/latent/`

1. **MSAModuleStack** (`msa_module.py`)
   - Stack of MSA blocks for sequence alignment processing
   - Row/column attention, outer product, transitions

2. **PairFormerStack** (`pairformer.py`)
   - Processes pair representations (residue-pair interactions)
   - Uses triangle attention + multiplicative updates
   - Supports kernel acceleration

3. **EvoformerBlock** (`evoformer.py`)
   - Combined MSA + Pair processing
   - CPU-GPU offloading for inference (`_offload_inference`)
   - Configurable column attention

4. **TemplateEmbedderAllAtom** (`template_module.py`)
   - Encodes structural templates
   - Generates pair biases from templates

### Structure Generation

#### Diffusion Module (12.5K LOC)
**File**: `openfold3/core/model/structure/diffusion_module.py`

**Key Components**:
1. **Noise Schedule Generation**
   - `create_noise_schedule()`: AF3 Protocol (Page 24)
   - Configurable sigma_data, s_max, s_min parameters

2. **Rotations & Augmentation**
   - `sample_rotations()`: Random quaternion sampling
   - `centre_random_augmentation()`: Noise injection (Algorithm 19)

3. **DiffusionModule Class**
   - Conditional diffusion process
   - Multi-step sampling with guided denoising
   - Produces final all-atom coordinates

#### Prediction Heads
**File**: `openfold3/core/model/heads/head_modules.py`
- **AuxiliaryHeadsAllAtom**: Confidence predictions
  - pLDDT (predicted local distance difference test)
  - PAE (predicted aligned error)
  - PDE (predicted distance error)
  - Experimentally resolved probability

### Input Processing

**File**: `openfold3/core/model/feature_embedders/input_embedders.py`
- **InputEmbedderAllAtom**: Encodes input sequences and features
- **MSAModuleEmbedder**: Processes MSA into latent representation

---

## 4. EXISTING MLX INTEGRATION

**Status**: NONE

No existing MLX code found in the repository. The codebase is purely PyTorch + CUDA.

---

## 5. BUILD SYSTEM AND DEPENDENCY MANAGEMENT

### Package Configuration (`pyproject.toml`)

**Direct Dependencies**:
```
Core: torch, pytorch-lightning, numpy, pandas, scipy, tqdm
ML: ml-collections, wandb
Structure: biotite >=1.1.0, rdkit, pdbeccdutils
Distributed: deepspeed, pytorch-lightning
GPU: NVIDIA-specific
  - cutlass <4 (NVIDIA CUTLASS kernels)
  - cuda-python <12.9.1
  - cuEquivariance (optional)
```

**Optional Dependencies**:
```
[cuequivariance]
- cuequivariance>=0.6.1
- cuequivariance-ops-torch-cu12>=0.6.1
- cuequivariance-torch>=0.6.1
- torch>=2.7
```

### Environment Setup
**Files**: `environments/production.yml`, `environments/development.txt`
- Mamba/Conda based
- CUDA 12.1.1 base
- cuDNN8 support
- MPI support (libopenmpi-dev)

### Docker Configuration
- **Standard Dockerfile**: NVIDIA CUDA 12.1.1 + cuDNN8
- **Dockerfile.blackwell**: NVIDIA CUDA 12.8 for Blackwell GPUs
- Multi-stage build with development + runtime images
- CUTLASS pre-installation
- DeepSpeed op pre-compilation hooks (commented out)

### Installation Scripts
**File**: `scripts/install_third_party_dependencies.sh`
- Installs CUTLASS
- Configures CUDA toolkit paths
- Sets environment variables (CUTLASS_PATH, CPATH)

---

## 6. ENTRY POINTS AND EXECUTION FLOW

### CLI Commands (run_openfold.py)

#### Training Pipeline
```python
run_openfold train --runner_yaml <config.yml> [--seed <N>] [--data_seed <N>]
```
- Loads `TrainingExperimentConfig` from YAML
- Creates `TrainingExperimentRunner`
- Calls: setup() → run()

**Execution Flow**:
1. `_torch_gpu_setup()`: GPU optimization setup
2. Config validation and creation
3. Runner initialization with DeepSpeed integration
4. Distributed training across GPUs

#### Inference Pipeline
```python
run_openfold predict --query_json <queries.json> [--inference_ckpt_path <ckpt>] \
                     [--num_diffusion_samples <N>] [--num_model_seeds <N>] \
                     [--runner_yaml <config.yml>] [--use_msa_server] [--use_templates] \
                     [--output_dir <dir>]
```

**Execution Flow**:
1. `_torch_gpu_setup()`: GPU optimization
2. Load `InferenceExperimentConfig`
3. Load queries from JSON
4. Create `InferenceExperimentRunner`
5. MSA computation (with optional server)
6. Template alignment (optional)
7. Model forward pass with diffusion sampling
8. Output generation

#### MSA Alignment Only
```python
run_openfold align-msa-server --query_json <queries.json> --output_dir <dir>
```

### Core Runner Classes

**File**: `openfold3/core/runners/model_runner.py`
- Abstract base class for all runners
- Model + config management
- Forward/backward pass coordination

**File**: `openfold3/projects/of3_all_atom/runner.py` (OpenFold3AllAtom)
- `setup(stage)`: Metric initialization, parameter freezing
- Training metrics: losses + correlation metrics
- Validation metrics: comprehensive evaluation
- `training_step()`: Loss computation + metric updates
- `validation_step()`: Full evaluation with model selection
- `_compute_losses()`: Orchestrates loss calculation
- `reseed()`: Reproducibility control

**File**: `openfold3/entry_points/experiment_runner.py`
- `TrainingExperimentRunner`: Full training setup with DeepSpeed
- `InferenceExperimentRunner`: Inference orchestration
- Model loading/downloading from checkpoint
- Batch processing and output serialization

### Model Flow (OpenFold3 Main Class)

**File**: `openfold3/projects/of3_all_atom/model.py` (27.7K LOC)

**Forward Pass** (Algorithm 1 from AF3):
1. **Input Embeddings**
   - `InputEmbedderAllAtom()`: Encode tokens + features
   - Dimension: `c_s` (sequence) and `c_z` (pair)

2. **MSA Processing**
   - `MSAModuleStack()`: 4 blocks of MSA attention + transitions
   - Produces refined MSA representation

3. **Pair-Former Stack**
   - `PairFormerStack()`: Multiple blocks
   - Processes pair (residue-residue) representations
   - Uses triangular attention + multiplicative updates
   - Optional kernel acceleration (DeepSpeed/cuEq)

4. **Diffusion Module**
   - `DiffusionModule()`: Structure generation
   - Iterative denoising from noise schedule
   - Produces atom coordinates

5. **Confidence Heads**
   - `AuxiliaryHeadsAllAtom()`: Prediction heads
   - pLDDT, PAE, PDE, resolution scoring

**Key Design Patterns**:
- Chunking for memory efficiency: `chunk_size` parameter
- Activation checkpointing: `blocks_per_ckpt`
- Kernel selection: `use_deepspeed_evo_attention`, `use_cueq_triangle_kernels`, `use_lma`
- Offloading: `_offload_inference` for CPU-GPU transfer during inference
- Masking: Per-token and per-pair masks for variable-length sequences

---

## 7. CRITICAL AREAS FOR MLX PORTING

### Priority 1: Core Tensor Operations
1. **Attention Mechanisms**
   - Replace `torch.einsum()` with MLX equivalent
   - Implement softmax with bias/mask support
   - Handle multi-head reshaping
   - ~100-150 LOC per variant

2. **Linear Transformations**
   - All `nn.Linear` layers → MLX linear ops
   - Custom weight initialization preservation
   - Gating mechanisms

3. **Activation Functions**
   - SwiGLU: `x1 * (x2 sigmoid_like_activation)`
   - LayerNorm with epsilon
   - ReLU, GELU variants

### Priority 2: Compute Kernels
1. **Softmax Optimization**
   - Triton kernels → MLX native ops (+ optimization)
   - With mask and bias support
   - Gradient computation for backward pass

2. **SwiGLU Fusion**
   - LigerSiLUMulFunction → MLX fused kernel
   - Element-wise operations

3. **Outer Product Operations**
   - `torch.einsum()` patterns → MLX einsum or explicit ops

### Priority 3: Structural Components
1. **Checkpointing**
   - Replace `torch.utils.checkpoint` with MLX equivalent
   - DeepSpeed checkpointing → standard MLX approach

2. **Mixed Precision**
   - `torch.amp.autocast()` → MLX dtype management
   - bfloat16 vs float32 handling

3. **Device Management**
   - Remove `torch.cuda.*` calls
   - CPU-GPU offloading → MLX device transfers

### Priority 4: Data Pipeline
1. **Batch Processing**
   - Tensor reshaping and masking operations
   - Variable-length sequence handling

2. **Feature Engineering**
   - Input embeddings (token encoding, feature projection)
   - MSA processing functions

3. **Output Generation**
   - Structure serialization
   - Confidence metric computation

### Priority 5: Training Infrastructure
1. **Loss Computation** (8 separate loss functions)
2. **Gradient Accumulation**
3. **Learning Rate Scheduling**
4. **Metric Computation**
5. **Checkpoint Management**

---

## 8. SPECIFIC CUDA PATTERNS TO REPLACE

### Pattern 1: Device Placement
```python
# Current CUDA pattern
model.cuda()
tensor.cuda()
.to(device="cuda")

# MLX approach
# Models/arrays already on GPU by default in MLX
```

### Pattern 2: Automatic Mixed Precision
```python
# Current
with torch.amp.autocast("cuda", enabled=False):
    result = operation()

# MLX approach
# Use mx.float32 / mx.float16 / mx.bfloat16 directly
result = mx.asarray(operation(), dtype=mx.float32)
```

### Pattern 3: Memory Management
```python
# Current
torch.cuda.empty_cache()
input_tensors[1] = input_tensors[1].cpu()

# MLX approach
# Memory auto-managed; explicit CPU transfer via:
mx.eval(array)  # Force evaluation
```

### Pattern 4: Kernel Dispatch
```python
# Current: Optional kernel usage
if use_deepspeed_evo_attention:
    output = DS4Sci_EvoformerAttention()
elif use_cueq_triangle_kernels:
    output = triangle_attention()
else:
    output = standard_attention()

# MLX approach
# Single optimized attention implementation
output = mlx_attention()
```

### Pattern 5: einsum Operations
```python
# Current
result = torch.einsum("...qc, ...kc->...qk", q, k)

# MLX approach
result = mx.einsum("...qc, ...kc->...qk", q, k)
# or explicit: matmul, reshape combinations
```

---

## 9. MODEL CONFIGURATION

### Reference Configuration
**File**: `openfold3/projects/of3_all_atom/config/model_config.py`

**Key Settings**:
- Architecture dimensions (hidden sizes, heads, blocks)
- Memory efficiency options:
  - `blocks_per_ckpt`: Checkpoint frequency
  - `use_deepspeed_evo_attention`: DeepSpeed kernel
  - `use_cueq_triangle_kernels`: cuEq kernel
  - `use_lma`: Low-memory attention
- Diffusion parameters (noise schedule, steps)
- Loss weights for multi-task learning
- Model selection metrics for validation

### Inference Configuration
**File**: `openfold3/projects/of3_all_atom/config/inference_query_format.py`
- JSON-based input specification
- Supports: proteins, RNAs, DNAs, ligands
- Template inclusion
- MSA pre-computation

---

## 10. TESTING INFRASTRUCTURE

**Files**: `openfold3/tests/test_*.py` (multiple test modules)

**Coverage**:
- `test_kernels.py`: Kernel correctness (DeepSpeed, cuEq, Triton)
- `test_of3_model.py`: Full model forward/backward passes
- `test_primitives.py`: Individual layer testing
- `test_inference_full.py`: End-to-end inference
- `test_model_runner.py`: Runner class validation

**Key Test Utilities**:
- `compare_utils.py`: CUDA availability checks, tolerance comparisons
- `data_utils.py`: Random input generation
- Fixtures for reproducible testing

---

## 11. SUMMARY: MIGRATION ROADMAP

### Phase 1: Foundation (Essential for any output)
1. Core tensor operations (linear, reshape, transpose)
2. Attention mechanisms (Q, K, V projections, softmax)
3. Activations and normalizations
4. Basic model structure (sequential blocks)

### Phase 2: Compute Efficiency
1. Einsum operations
2. Softmax with bias/mask
3. Fused operations (SwiGLU, etc.)
4. Optional: Kernel optimization

### Phase 3: Model Completeness
1. Full forward pass (all attention variants)
2. Diffusion module
3. Loss computation (training)
4. Metrics and validation

### Phase 4: Features & Polish
1. Checkpointing/memory optimization
2. Multi-sample generation
3. Output serialization
4. Configuration management

### Phase 5: Production
1. Performance optimization
2. Apple Silicon-specific tuning
3. Comprehensive testing
4. Documentation

---

## 12. DEPENDENCY IMPLICATIONS FOR MLX

### Libraries to Replace
- **torch** → mlx.core
- **torch.nn** → mlx.nn (or custom layers)
- **deepspeed** → MLX native optimization (or remove)
- **triton** → MLX native ops (remove)
- **cuequivariance** → MLX native ops (remove)
- **pytorch_lightning** → MLX/custom training loop
- **torchmetrics** → Custom metric implementations

### Libraries to Keep/Adapt
- **biotite**: Structure I/O (no GPU acceleration)
- **rdkit**: Molecular processing (no GPU acceleration)
- **ml-collections**: Configuration management (pure Python)
- **pdbeccdutils**: Ligand processing (pure Python)
- **scipy, numpy**: Can use MLX arrays where needed

### New Dependencies to Add
- **mlx**: Main framework
- **mlx-data**: Data pipeline (optional)
- **numpy**: NumPy compatibility layer (if needed)

---

## CONCLUSION

OpenFold3 is a **sophisticated, well-structured PyTorch model** with multiple GPU acceleration pathways (DeepSpeed, cuEquivariance, Triton). The codebase is modular, making systematic porting to MLX feasible. The main challenges will be:

1. Replacing 3 specialized GPU kernels with MLX native ops
2. Adapting mixed-precision training without torch.amp
3. Maintaining numerical precision (especially for structures)
4. Preserving performance on Apple Silicon

The core computational logic is straightforward and doesn't rely on obscure PyTorch internals, which is favorable for porting.

