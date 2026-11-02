# Smooth lDDT ball query

This page records the protein–RNA study behind the reweighted Triton ball-query
smooth lDDT loss. How to turn the backend on is in {doc}`Kernels <kernels>`.

The study has two parts:

- **RTX 4090:** the reweighted reduction against a matched unweighted reduction
  on the same current compacted kernel. This isolates the cost of reweighting.
- **H800:** the current implementation (`778f30aa`) against the historical
  ball-query implementation (`074d75c2`). This includes kernel and compaction
  changes as well as reweighting, so it is not a reweighting-only measurement.

At `K=512`, type-aware reweighting improves gradient agreement with dense
smooth lDDT on the 50 structures both machines could score: relative L2
`0.26738 → 0.25021`, cosine `0.95665 → 0.96995`.

---

## 1. Background and motivation

The default dense smooth-lDDT loss constructs pairwise atom-distance data with
quadratic `O(A²)` scaling for `A` resolved atoms. This gives an exact reference,
but its memory cost becomes prohibitive for large all-atom structures.

The Triton ball-query backend instead scans candidates and retains at most `K`
in-radius neighbors for each center. Its dominant saved neighborhood storage is
therefore `O(AK)` rather than `O(A²)`.

The two center types use different physical cutoffs:

- non-nucleotide centers: **15 Å**;
- nucleotide centers: **30 Å**.

A 30 Å sphere has eight times the volume of a 15 Å sphere under a uniform-density
approximation. Using one fixed reservoir size therefore retains a smaller
fraction of an RNA-centered neighborhood than of a protein-centered
neighborhood. A simple unweighted truncated reduction can consequently
underrepresent nucleotide-centered rows.

The reweighted implementation corrects this imbalance for truncated rows while
preserving the normalization of the smooth-lDDT reduction. It applies the same
weight to each row's score sum and pair-count denominator.

For reservoir size `K`, the lower-bounded row weights are:

```text
non-nucleotide weight = max(1,  512 / K)
nucleotide weight     = max(1, 2048 / K)
```

The tested schedule is:

| K | Non-nucleotide weight | Nucleotide weight |
|---:|---:|---:|
| 256 | 2 | 8 |
| 512 | 1 | 4 |
| 1024 | 1 | 2 |
| 2048 | 1 | 1 |

Rows whose complete neighborhood fits in the reservoir retain weight one. The
additional weighted reduction uses `O(A)` metadata and does not add another
atom-by-`K` tensor.

---

## 2. Implementations under test

### 2.1 Dense reference

The dense reference is the default `smooth_lddt_loss`. It computes the complete
pairwise neighborhood and serves as the numerical target for scalar loss and
predicted-coordinate gradients.

### 2.2 Current reweighted ball query

The current implementation is commit:

```text
778f30aad9144dea57f7cee879e60b7418de71b8
feat: add reweighted ball-query smooth lDDT
```

It includes type-aware row reweighting and the current compacted ball-query
kernel/reduction implementation.

### 2.3 Matched current unweighted reduction

The RTX 4090 study disables row reweighting while retaining the current query,
compaction, and reduction machinery. This is the correct controlled comparison
for answering:

> What numerical and performance change comes from adding reweighting to the
> current implementation?

### 2.4 Exact historical ball query

The H800 study loads the exact historical implementation from commit:

```text
074d75c2
feat: wire Triton smooth lDDT into diffusion loss
```

It is executed from an isolated source snapshot in a fresh process, rather than
through the compatibility function retained in current code. This is the
correct comparison for answering:

> How does the current implementation behave relative to the actual original
> implementation before the reweighting update?

Because this historical comparison also includes intervening kernel and
compaction changes, it cannot isolate the cost of reweighting by itself.

---

## 3. Benchmark dataset

Both studies use the same pinned 51-structure protein–RNA validation set.

| Property | Value |
|---|---|
| Structures | 51 unique PDB entries, each present once |
| Representation | Every resolved atom; no spatial cropping |
| Composition | Every case contains protein and nucleotide atoms |
| Atom range | 666–20,016 |
| Coordinate perturbation | Independent Gaussian noise, σ = 1 Å |
| Large-structure ladder | `7vtn`, `7r9f`, `7v93`, `8h2h`, `7ozs`, `7r7c` |

The dataset setup verifies:

- exact PDB membership;
- one case per structure;
- no atom-count reduction relative to the source structure;
- 1,785 required structure, alignment, and template files;
- 47 reference-molecule entries.

The six large cases provide the strongest scaling test. The largest,
`8h2h`, contains 20,016 resolved atoms and is the only case on which the RTX
4090 dense reference runs out of memory.

---

## 4. Shared benchmark protocol

| Setting | Value |
|---|---|
| dtype | FP32 |
| Protein radius | 15 Å |
| Nucleotide radius | 30 Å |
| K | 256, 512, 1024, 2048 |
| Reservoir seeds | 0, 1, 2 |
| Warmups | 5 per backend block |
| Timed repetitions | 20 per backend block |
| Training measurement | Forward and backward |
| Inference measurement | Forward-only under inference mode |
| Timing statistic | Median CUDA-event time per case/seed block |
| Report aggregation | Mean of per-case, per-seed median timings |
| Memory | Incremental peak allocated CUDA memory above resident inputs |

For accuracy, each sparse scalar loss and predicted-coordinate gradient is
compared with dense smooth lDDT computed on the same structure and perturbed
coordinates.

The reported metrics are:

- absolute scalar-loss error;
- gradient relative L2 error;
- gradient cosine similarity;
- maximum absolute gradient difference in raw artifacts;
- training runtime;
- forward-only runtime;
- incremental peak training memory;
- incremental peak forward memory.

---

## 5. Hardware and software environments

| Property | RTX 4090 study | H800 study |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090 | NVIDIA H800 |
| Available memory | approximately 24 GiB | 81,559 MiB |
| Driver | recorded in original run artifacts | 550.144.03 |
| PyTorch | 2.5.1 | 2.5.1 |
| Triton | 3.1.0 | 3.1.0 |
| CUDA build | 12.4 | 12.4 |
| Sparse comparison | matched current unweighted | exact historical commit |
| Dense coverage | 50/51 structures | 51/51 structures |

### H800 isolation controls

The H800 host had two GPUs. One was occupied throughout the campaign and was
excluded. All measurements used only:

```text
GPU-6a04420a-ab52-b668-87ae-be5bbd2ca6fd
```

The campaign enforced:

- one benchmark process at a time;
- sequential backend/K blocks;
- an exclusive campaign lock;
- empty-GPU checks before and after each block;
- process and utilization sampling every two seconds;
- termination if a foreign compute process appeared;
- separate Triton caches for current and historical implementations;
- fresh processes to avoid Python module or compiled-kernel contamination.

No overlap was detected. The selected GPU returned to 4 MiB used and 0%
utilization after the campaign.

---

## 6. Validation coverage

### RTX 4090

The four K runs produced 1,428 result rows. All 1,224 sparse rows completed.
Dense completed 50 structures and OOMed only on `8h2h`.

### H800

The four K runs also produced 1,428 result rows:

- 612 current reweighted rows;
- 612 exact historical ball-query rows;
- 204 dense rows;
- 0 failed rows;
- 0 OOM rows.

The H800 therefore adds a dense reference for `8h2h` while reproducing the
50-case numerical comparison used by the RTX report.

---

## 7. Cross-hardware numerical reproducibility

The following table uses the 50-structure common subset because `8h2h` lacks a
dense reference on the RTX 4090. The values from both runs agree to the shown
precision.

| K | Unweighted/original loss error | Reweighted loss error | Unweighted/original gradient L2 | Reweighted gradient L2 | Unweighted/original cosine | Reweighted cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 5.089e-4 | **3.408e-4** | 0.35873 | **0.25090** | 0.92736 | **0.96906** |
| 512 | **3.488e-4** | 4.278e-4 | 0.26738 | **0.25021** | 0.95665 | **0.96995** |
| 1024 | **1.444e-4** | 1.466e-4 | 0.12543 | **0.12198** | 0.98697 | **0.99096** |
| 2048 | 1.70e-5 | 1.71e-5 | 0.02398 | 0.02398 | 0.99884 | 0.99884 |

### Interpretation

- At `K=256`, reweighting improves scalar loss and both gradient metrics.
- At `K=512`, reweighting slightly worsens mean scalar-loss error but improves
  both gradient metrics.
- At `K=1024`, scalar loss is nearly unchanged while both gradient metrics
  improve.
- At `K=2048`, both weights equal one; differences reduce to floating-point
  order.

For training, gradient agreement is the more directly relevant signal. The
scalar loss and its gradient measure different properties, so the small scalar
regressions at `K=512` and `1024` should not be hidden or conflated with the
clear gradient improvement.

### H800 all-51 accuracy

With the H800 dense reference for `8h2h`, the complete 51-case means are:

| K | Exact original gradient L2 | Reweighted gradient L2 | Exact original cosine | Reweighted cosine |
|---:|---:|---:|---:|---:|
| 256 | 0.36025 | **0.25152** | 0.92682 | **0.96887** |
| 512 | 0.26947 | **0.25058** | 0.95608 | **0.96981** |
| 1024 | 0.12824 | **0.12406** | 0.98651 | **0.99062** |
| 2048 | 0.02596 | 0.02596 | 0.99871 | 0.99871 |

The complete-set result supports the same conclusion as the common subset.

---

## 8. Runtime on RTX 4090

The RTX runtime table is the controlled current-kernel comparison and therefore
the primary estimate of isolated reweighting overhead.

| K | Dense train | Matched unweighted train | Reweighted train | Dense forward | Matched unweighted forward | Reweighted forward |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16.351 ms | 10.244 ms | 10.556 ms | 8.389 ms | 2.941 ms | 2.889 ms |
| 512 | 16.536 ms | 10.721 ms | 11.056 ms | 8.429 ms | 2.717 ms | 2.679 ms |
| 1024 | 16.419 ms | 10.850 ms | 11.187 ms | 8.425 ms | 2.572 ms | 2.537 ms |
| 2048 | 16.177 ms | 12.148 ms | 12.489 ms | 8.345 ms | 2.751 ms | 2.811 ms |

On RTX 4090:

- reweighted training is 2.8–3.1% slower than the matched unweighted current
  reduction;
- the absolute difference is approximately 0.31–0.34 ms;
- forward-only timing has no consistent penalty and remains within 2.2%;
- both sparse backends are faster than dense over the complete executable set.

This small training difference is consistent with the extra `O(A)` type-group
row reduction.

---

## 9. Runtime on H800

The H800 table compares the exact historical implementation with the current
reweighted implementation over all 51 cases.

| K | Dense train | Exact original train | Reweighted train | Dense forward | Exact original forward | Reweighted forward |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8.584 ms | 8.805 ms | 9.371 ms | 4.946 ms | 3.167 ms | 3.544 ms |
| 512 | 8.552 ms | 8.383 ms | 8.931 ms | 4.959 ms | 2.501 ms | 2.761 ms |
| 1024 | 8.555 ms | 7.722 ms | 8.776 ms | 4.984 ms | 2.104 ms | 2.346 ms |
| 2048 | 8.374 ms | 8.848 ms | 9.325 ms | 4.967 ms | 1.778 ms | 1.846 ms |

Relative to the exact historical commit, the current reweighted implementation
is:

| K | Training difference | Forward difference |
|---:|---:|---:|
| 256 | +6.4% | +11.9% |
| 512 | +6.5% | +10.4% |
| 1024 | +13.7% | +11.5% |
| 2048 | +5.4% | +3.8% |

These percentages include all historical-to-current implementation changes.
They are not estimates of reweighting-only overhead. For that question, use the
RTX matched-current comparison above.

---

## 10. Cross-hardware runtime comparison

### 10.1 Dense

On the RTX-common 50 structures, H800 dense training averages 6.35–6.57 ms,
compared with 16.18–16.54 ms on RTX 4090. This is a reduction of approximately
60%.

Dense forward averages 3.72–3.76 ms on H800 versus 8.35–8.43 ms on RTX 4090,
roughly 55% lower.

### 10.2 Current reweighted sparse backend

Across all 51 H800 cases versus the corresponding RTX aggregate:

| K | RTX reweighted train | H800 reweighted train | Change | RTX reweighted forward | H800 reweighted forward | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 10.556 ms | 9.371 ms | −11.2% | 2.889 ms | 3.544 ms | +22.7% |
| 512 | 11.056 ms | 8.931 ms | −19.2% | 2.679 ms | 2.761 ms | +3.1% |
| 1024 | 11.187 ms | 8.776 ms | −21.5% | 2.537 ms | 2.346 ms | −7.5% |
| 2048 | 12.489 ms | 9.325 ms | −25.3% | 2.811 ms | 1.846 ms | −34.3% |

Sparse forward timing is more launch- and shape-sensitive than dense timing.
The H800 advantage becomes clearer as K increases and each launch performs more
work. At low K, fixed launch overhead and aggregation across many small cases
can dominate.

### 10.3 Scope warning

The RTX and H800 runtime rows are not perfectly symmetric experimental objects:

- the RTX study's unweighted backend is the current matched kernel;
- the H800 study's original backend is historical code;
- H800 aggregates all 51 structures, including `8h2h`, unless stated otherwise;
- RTX dense aggregates only 50 structures.

Numerical comparisons use the common subset where needed. Runtime tables retain
each machine's complete successful coverage and explicitly state that scope.

---

## 11. RTX 4090 memory

The matched current unweighted and reweighted reductions have identical peak
memory in all 612 paired RTX comparisons. This establishes that reweighting
does not add another atom-by-`K` allocation.

| K | Sparse train mean/max | Sparse forward mean/max | Dense train mean/max | Dense forward mean/max |
|---:|---:|---:|---:|---:|
| 256 | 53.89 / 310.62 MiB | 36.54 / 210.56 MiB | 1140.21 / 12836.69 MiB | 569.53 / 6418.59 MiB |
| 512 | 108.12 / 618.58 MiB | 73.25 / 418.52 MiB | 1140.21 / 12836.69 MiB | 569.53 / 6418.59 MiB |
| 1024 | 213.84 / 1212.75 MiB | 144.87 / 821.75 MiB | 1140.21 / 12836.69 MiB | 569.53 / 6418.59 MiB |
| 2048 | 406.14 / 2424.66 MiB | 275.12 / 1642.72 MiB | 1140.21 / 12836.69 MiB | 569.53 / 6418.59 MiB |

Sparse maxima include `8h2h`; dense maxima do not, because RTX dense OOMs on
that case.

---

## 12. H800 memory

The H800 exact-historical comparison shows the effect of the complete current
implementation, including compaction changes.

| K | Reweighted train mean/max | Exact original train mean/max | Reweighted forward mean/max | Exact original forward mean/max |
|---:|---:|---:|---:|---:|
| 256 | 53.89 / 310.62 MiB | 59.17 / 340.39 MiB | 36.54 / 210.56 MiB | 48.76 / 280.62 MiB |
| 512 | 108.12 / 618.58 MiB | 118.42 / 673.59 MiB | 73.25 / 418.52 MiB | 97.63 / 554.73 MiB |
| 1024 | 213.84 / 1212.75 MiB | 234.18 / 1330.03 MiB | 144.87 / 821.75 MiB | 193.05 / 1095.70 MiB |
| 2048 | 406.14 / 2424.66 MiB | 445.12 / 2659.67 MiB | 275.12 / 1642.72 MiB | 366.63 / 2190.78 MiB |

Relative to the exact historical implementation, the current implementation
uses approximately:

- 8.8–9.0% less incremental training memory;
- 25% less incremental forward memory.

This does not conflict with the RTX finding that matched current weighted and
unweighted reductions use identical memory. The two statements answer
different questions:

- row reweighting itself adds no measurable peak allocation;
- the broader current implementation is more compact than the old commit.

---

## 13. Dense memory and the `8h2h` case

The H800 completes dense smooth lDDT on all structures. Across all 51 cases,
dense incremental memory is:

- training: **1597.34 MiB mean / 24453.83 MiB max**;
- forward: **798.11 MiB mean / 12227.26 MiB max**.

The maxima come from `8h2h`. On that structure alone:

| Backend/configuration | Train time | Forward time | Train peak | Forward peak |
|---|---:|---:|---:|---:|
| Dense | ~109.3 ms | ~66.2 ms | 24453.83 MiB | 12227.26 MiB |
| Reweighted K=256 | 33.35 ms | 28.06 ms | 310.62 MiB | 210.56 MiB |
| Reweighted K=512 | 32.23 ms | 25.05 ms | 618.58 MiB | 418.52 MiB |
| Reweighted K=1024 | 34.10 ms | 20.64 ms | 1212.75 MiB | 821.75 MiB |
| Reweighted K=2048 | 36.47 ms | 9.88 ms | 2424.66 MiB | 1642.72 MiB |

At the default `K=512`, sparse training uses about **39.5× less incremental
memory** than dense on `8h2h`, while completing substantially faster.

The RTX 4090 OOM is therefore expected: the measured H800 incremental dense
training peak alone is approximately 23.9 GiB, before accounting for resident
inputs, runtime context, allocator fragmentation, and other process memory.

---

## 14. Large-structure behavior

### 14.1 RTX 4090 large cases

The original RTX table reports reweighted accuracy over the five large cases
with dense references and runtime/memory over all six sparse cases.

| K | Reweighted loss error | Gradient L2 | Cosine | Train | Forward | Train/forward max memory |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.003e-4 | 0.27043 | 0.96297 | 13.379 ms | 7.043 ms | 310.62 / 210.56 MiB |
| 512 | 1.857e-4 | 0.22982 | 0.97270 | 13.971 ms | 6.189 ms | 618.58 / 418.52 MiB |
| 1024 | 1.232e-4 | 0.18106 | 0.98359 | 17.159 ms | 6.322 ms | 1212.75 / 821.75 MiB |
| 2048 | 8.049e-5 | 0.12761 | 0.99213 | 26.260 ms | 7.639 ms | 2424.66 / 1642.72 MiB |

### 14.2 H800 large cases

The H800 table includes dense accuracy for all six cases.

| K | Reweighted loss error | Gradient L2 | Cosine | Train | Forward | Train/forward max memory |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.836e-4 | 0.27244 | 0.96235 | 16.694 ms | 12.117 ms | 310.62 / 210.56 MiB |
| 512 | 1.681e-4 | 0.23641 | 0.97109 | 14.654 ms | 9.074 ms | 618.58 / 418.52 MiB |
| 1024 | 1.187e-4 | 0.18890 | 0.98193 | 14.993 ms | 7.266 ms | 1212.75 / 821.75 MiB |
| 2048 | 7.431e-5 | 0.12714 | 0.99217 | 16.958 ms | 4.858 ms | 2424.66 / 1642.72 MiB |

The accuracy difference between these tables primarily reflects inclusion of
`8h2h` in the H800 dense-reference set. Memory maxima are hardware-independent
allocator results and match across the current sparse implementation.

---

## 15. Scaling analysis

### 15.1 Sparse memory scales with K

Doubling K approximately doubles sparse memory:

- K=256: 53.89 MiB mean training memory;
- K=512: 108.12 MiB;
- K=1024: 213.84 MiB;
- K=2048: 406.14 MiB.

The relationship is not perfectly proportional because resident inputs and
`O(A)` metadata do not scale with K, but the dominant reservoir allocation does.

### 15.2 Dense memory scales with atom-pair count

Dense memory is independent of K and strongly dependent on structure size. Its
maximum rises from 12.84 GiB on the RTX-executable subset to 24.45 GiB when the
20,016-atom case is included.

### 15.3 Accuracy improves with K

Increasing K reduces truncation:

- gradient relative L2 falls from approximately 0.25 at K=256/512 to 0.122 at
  K=1024 and 0.024 at K=2048 for the reweighted common subset;
- cosine rises from approximately 0.969 to 0.991 and then 0.999;
- memory rises approximately linearly.

K therefore exposes a direct memory-versus-fidelity control.

---

## 16. Recommended default and operating points

### K=512: recommended default

`K=512` remains the recommended default when training memory and throughput are
important:

- it retains the intended 1:4 non-nucleotide/nucleotide correction;
- it improves gradient agreement over unweighted/original K=512;
- it uses about 108 MiB mean and 619 MiB maximum incremental training memory
  over this demanding 51-case set;
- isolated reweighting overhead on RTX 4090 is approximately 3%;
- it completes the 20,016-atom case with only 619 MiB incremental training
  allocation on H800.

### K=1024: higher-fidelity middle ground

Choose `K=1024` when roughly doubling K=512 reservoir memory is acceptable:

- gradient relative L2 improves from 0.250 to 0.122;
- cosine improves from 0.970 to 0.991;
- mean training memory rises from 108 to 214 MiB.

### K=2048: near-dense gradients

Choose `K=2048` when near-dense gradients justify the memory cost:

- relative L2 is approximately 0.024;
- cosine is approximately 0.999;
- both type weights equal one, so the correction is inactive;
- mean training memory is about 406 MiB and the largest case uses 2.42 GiB.

### K=256: minimum-memory mode

`K=256` minimizes reservoir memory and benefits most strongly from reweighting,
but gradient approximation remains less accurate than at larger K.

---

## 17. Conclusions

The combined RTX 4090 and H800 evidence supports the following conclusions.

1. **The numerical benefit is reproducible.** The H800 reproduces the RTX
   4090 accuracy table to displayed precision on the common 50 structures.
2. **Reweighting improves the training signal.** Gradient relative L2 and
   cosine improve at K=256, 512, and 1024.
3. **The scalar-loss trade-off is small and explicit.** Mean scalar error
   improves at K=256 but is slightly worse at K=512 and K=1024.
4. **Reweighting itself has low overhead.** On the controlled matched-current
   RTX comparison, training overhead is approximately 3% and forward overhead
   is inconsistent/negligible.
5. **Reweighting adds no atom-by-K memory allocation.** Matched current weighted
   and unweighted peaks are identical in all 612 RTX pairs.
6. **The current implementation is more compact than the historical one.** The
   H800 exact-commit comparison shows about 9% lower training memory and 25%
   lower forward memory.
7. **Sparse scaling removes the dense memory barrier.** Every sparse case
   completes on both GPUs. Dense fails on the largest case on RTX 4090 and
   consumes 24.45 GiB incremental training memory on H800.
8. **K=512 is a practical default.** It provides improved gradient fidelity,
   low overhead, and a large memory reduction while preserving configurability
   for higher-fidelity K values.

Overall, the reweighted ball-query smooth-lDDT backend provides a strong
replacement for dense smooth lDDT in memory-constrained all-atom training. It
preserves controllable fidelity, improves the bias introduced by a shared
truncated reservoir, and remains practical from RTX-class accelerators through
large-memory data-center GPUs.

---

## 18. Artifacts

The tables above are aggregated from two campaigns that used the same 51
structures, the same coordinate perturbations, and the same dense smooth-lDDT
definition.

Raw JSON, CSV, campaign logs, and the historical `074d75c2` source snapshot
are not part of this repository. They can be provided on request. The OpenFold3
checkout used for the runs was left clean.

