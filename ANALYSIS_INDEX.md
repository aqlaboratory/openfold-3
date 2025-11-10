# OpenFold3 MLX Porting - Comprehensive Analysis Index

This directory contains three comprehensive analysis documents to guide the porting of OpenFold3 from PyTorch/CUDA to MLX for Apple Silicon support.

## Documents Overview

### 1. CODEBASE_ANALYSIS.md (19 KB, 606 lines)
**Comprehensive technical deep-dive into the entire codebase**

Contents:
- Executive summary
- Overall project structure (250 files, ~72K LOC)
- GPU/CUDA dependencies (20 files with GPU refs)
  - DeepSpeed4Science (EvoformerAttention)
  - cuEquivariance (triangle kernels)
  - Triton kernels (softmax, SwiGLU)
- Key computational modules
  - Attention mechanisms (672 LOC)
  - Normalization/Activation layers
  - Structure generation (Diffusion module)
  - Model architecture stacks
- Existing MLX integration (none)
- Build system and dependencies
- Entry points and execution flow
- Critical areas for MLX porting (Priority 1-5)
- Specific CUDA patterns to replace (5 key patterns)
- Model configuration details
- Testing infrastructure
- Migration roadmap (5 phases)
- Dependency implications

**Best for**: Understanding the full scope of work, design decisions, and architectural patterns

---

### 2. QUICK_SUMMARY.txt (6.4 KB, 219 lines)
**Quick reference guide for rapid understanding and decision-making**

Contents:
- Project statistics
- Critical CUDA/GPU components (3 main kernels)
- Core computational modules overview
- GPU memory management patterns (3 key patterns)
- Dependency stack to replace vs. keep
- Entry points and execution flow (3 CLI commands)
- Porting priority roadmap (5 phases with timelines)
- Estimated effort breakdown
- Key challenges (5 major challenges)

**Best for**: Quick orientation, status updates, discussion, and sprint planning

---

### 3. FILE_STRUCTURE_REFERENCE.md (11 KB)
**Detailed file-by-file reference guide**

Contents:
- Entry points with line numbers
- Core model architecture organized by:
  - Main model class
  - Primitives (basic building blocks)
  - Attention variants
  - Latent representation stacks
  - Diffusion & structure
  - Input/output processing
- GPU-accelerated kernels (with fallback info)
- Training & inference infrastructure
- Loss functions and metrics
- Utility functions by category
- Configuration files
- Data processing
- Testing infrastructure
- Critical patterns by location (with line numbers)
- File dependency graph
- Summary statistics

**Best for**: Navigation, locating specific modules, understanding dependencies, implementation work

---

## Key Findings

### Project Scale
- **72,000 lines of code** across **250 Python files**
- Well-organized, modular structure
- Clear separation of concerns (primitives → layers → modules → model)

### GPU Acceleration Strategy
Three complementary acceleration approaches:
1. **DeepSpeed4Science**: Memory-efficient attention (optional, seq_len > 16)
2. **cuEquivariance**: Equivariant tensor operations (optional, shape-dependent)
3. **Triton**: Custom kernels for softmax, SwiGLU (optional, with PyTorch fallbacks)

All three have fallbacks to standard PyTorch, making removal feasible.

### Core Computations
- **Attention mechanisms** (672 LOC): Multi-head, triangular, MSA row/column
- **Diffusion module** (12.5K LOC): Structure generation with noise schedule
- **Normalization/Activation**: LayerNorm, AdaLN, SwiGLU
- **Linear layers**: Custom initializations per AF3 spec

### Porting Challenges

**Tier 1 (Must solve)**:
1. Replace DeepSpeed EvoformerAttention with MLX ops
2. Handle precision requirements (maintain bfloat16, float32 carefully)
3. Implement custom weight initializations

**Tier 2 (Important)**:
1. CPU-GPU offloading patterns for memory efficiency
2. Mixed-precision training without torch.amp
3. Checkpointing and gradient computation

**Tier 3 (Nice to have)**:
1. Performance parity with specialized kernels
2. Multi-GPU distributed training
3. Integration with DeepSpeed optimizations

---

## Recommended Reading Order

### For Decision-Making
1. Start with **QUICK_SUMMARY.txt** (10 mins)
2. Review **CODEBASE_ANALYSIS.md** sections:
   - Executive Summary
   - GPU/CUDA Dependencies (Section 2)
   - Critical Areas (Section 7)

### For Implementation Planning
1. Read **CODEBASE_ANALYSIS.md** sections:
   - Key Computational Modules (Section 3)
   - Entry Points (Section 6)
   - Migration Roadmap (Section 11)
2. Reference **FILE_STRUCTURE_REFERENCE.md** for:
   - Core Model Architecture
   - Critical Patterns by Location
   - File Dependency Graph

### For Actual Implementation
1. Use **FILE_STRUCTURE_REFERENCE.md** as your main guide
2. Cross-reference with **CODEBASE_ANALYSIS.md** for:
   - Detailed pattern descriptions
   - GPU operation specifics
   - Precision handling requirements
3. Check **QUICK_SUMMARY.txt** for:
   - Effort estimates
   - Phase timelines
   - Challenge documentation

---

## Key Metrics for Planning

### Effort Estimates
- **Phase 1 (Foundation)**: 5-7 days
- **Phase 2 (Efficiency)**: 3-5 days
- **Phase 3 (Completeness)**: 8-12 days
- **Phase 4 (Features)**: 5-7 days
- **Phase 5 (Polish)**: 5-10 days
- **Total**: 3-4 weeks for core porting + 1-2 weeks for optimization

### File Priority Ranking
**Must port first** (highest priority):
1. `openfold3/core/model/primitives/attention.py` (672 LOC)
2. `openfold3/core/model/primitives/linear.py` (~150 LOC)
3. `openfold3/core/model/primitives/normalization.py` (~150 LOC)
4. `openfold3/core/model/primitives/activations.py` (~60 LOC)

**Port early** (high priority):
1. `openfold3/core/model/layers/triangular_attention.py`
2. `openfold3/core/model/latent/msa_module.py`
3. `openfold3/core/model/latent/pairformer.py`
4. `openfold3/core/model/latent/evoformer.py`

**Port once core is working** (medium priority):
1. `openfold3/core/model/structure/diffusion_module.py` (12.5K LOC)
2. `openfold3/projects/of3_all_atom/model.py` (27.7K LOC - main model)
3. `openfold3/core/loss/` (all loss functions)

**Port last** (low priority):
1. Training infrastructure (runners, metrics)
2. Data pipeline
3. Distributed training support

---

## Critical Code Patterns to Know

### Pattern 1: Kernel Selection (30 lines)
**File**: `openfold3/core/model/primitives/attention.py` lines 352-372
```python
if cueq_would_fall_back(...):
    use_cueq_triangle_kernels = False
if use_deepspeed_evo_attention and q_x.shape[-2] <= 16:
    use_deepspeed_evo_attention = False
if use_cueq_triangle_kernels:
    o = _cueq_triangle_attn(q, k, v, biases, scale=scale)
elif use_deepspeed_evo_attention:
    o = _deepspeed_evo_attn(q, k, v, biases)
elif use_lma:
    o = _lma(q, k, v, biases, chunk_size_q, chunk_size_kv)
else:
    o = _attention(q, k, v, biases)
```
**MLX approach**: Single optimized implementation (conditional code can be removed)

### Pattern 2: Precision Handling (30 lines)
**File**: `openfold3/core/model/primitives/normalization.py` lines 59-86
```python
d = x.dtype
if d is torch.bfloat16 and not deepspeed_is_initialized:
    with torch.amp.autocast("cuda", enabled=False):
        # Compute in float32
else:
    # Compute in original dtype
```
**MLX approach**: Use `mx.asarray(..., dtype=mx.bfloat16)` directly

### Pattern 3: Attention Core (20 lines)
**File**: `openfold3/core/model/latent/evoformer.py` lines 94-135
```python
attn_dtype = torch.float32 if use_high_precision else query.dtype
with torch.amp.autocast("cuda", dtype=attn_dtype):
    scores = torch.einsum("...qc, ...kc->...qk", query, key)
    for b in biases:
        scores += b
    scores = softmax_no_cast(scores, dim=-1)
attention = torch.einsum("...qk, ...kc->...qc", scores.to(value.dtype), value)
```
**MLX approach**: Straightforward einsum and addition operations

---

## Next Steps

1. **Immediate**: Review this index and start with QUICK_SUMMARY.txt
2. **Planning**: Create task breakdown using FILE_STRUCTURE_REFERENCE.md
3. **Development**: Port primitives first (linear, attention, norm) using CODEBASE_ANALYSIS.md
4. **Testing**: Validate each component against original outputs
5. **Integration**: Assemble into full model following dependency graph

---

## Document Version Info

- **Generated**: 2025-11-10
- **Codebase Version**: OpenFold3-preview (0.3.1)
- **PyTorch Version Ref**: 2.5.1+
- **CUDA Version Ref**: 12.1.1+
- **Analysis Scope**: Full repository assessment

---

## Questions & Clarifications

**Q: Can we run the original PyTorch model alongside MLX during development?**
A: Yes, both can coexist. Recommended: Develop MLX in separate branch, compare outputs.

**Q: What's the minimum viable product?**
A: Inference with default settings (no specialized kernels) should work for initial porting.

**Q: Should we port training immediately?**
A: Focus on inference first (6-8 days), then add training infrastructure (5-7 days).

**Q: How do we validate numerical correctness?**
A: Compare outputs with PyTorch reference, use test fixtures provided in codebase.

**Q: Can we use PyTorch models as checkpoints?**
A: Yes, but will need weight format conversion (PyTorch dict → MLX arrays).

