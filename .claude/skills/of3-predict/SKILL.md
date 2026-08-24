---
name: of3-predict
description: "Run an OpenFold3 (OF3) structure prediction on a query.json via `run_openfold predict`, with guidance on common configuration toggles: ColabFold MSA server on/off, precomputed MSAs, templates on/off, seeds/diffusion samples, output format (cif/pdb), low-memory preset, multi-GPU, evoformer kernel choice (cuequivariance/triton), and pocket constraints. Use when asked to run/predict/fold a structure from a query.json, or to change how an OF3 prediction is configured. For building the query.json itself, use of3-build-query first."
---

# Running an OF3 prediction

One command, `run_openfold predict`, does everything. Configuration comes from three layers, later overrides earlier:

1. `$OPENFOLD_CACHE/runner.yml` (default `~/.openfold3/runner.yml`) — persistent user defaults, applied automatically if present.
2. `--runner-yaml <path>` — a YAML file for this run.
3. CLI flags (`--use-msa-server`, `--num-model-seeds`, ...) — highest precedence.

## 0. Prerequisites

- A `query.json` (see `of3-build-query` skill if it doesn't exist yet). No query yet and just want to smoke-test the command? `examples/example_inference_inputs/` has ready-to-run samples (`query_ubiquitin.json` is the smallest).
- An activated OF3 environment. If unsure which pixi environment to use, run `pixi environment list` (or check `pixi.toml`) in the repo root — GPU users on CUDA typically use `openfold3-cuda12`; CPU/macOS uses `openfold3-base`. Prefix commands with `pixi run -e <env>` if not already inside an activated shell.
- Model checkpoint: auto-downloaded to `$OPENFOLD_CACHE` (default `~/.openfold3/`) on first run if `--inference-ckpt-path`/`--inference-ckpt-name` aren't given.

## 1. Minimal command

```bash
run_openfold predict \
    --query-json /path/to/query.json \
    --output-dir /path/to/output/
```

Defaults: ColabFold MSA server on, no templates, 1 seed, 5 diffusion samples, `.cif` output.

## 2. Validate before a real run

Before committing to the user's actual (possibly large/expensive) job, it's cheap insurance to confirm the environment and the query are both sound. Two independent checks — offer both, they catch different things:

**a. Environment/install smoke test** — runs a small built-in case end-to-end and checks the numeric output against known-good ranges. Confirms the accelerator, checkpoint, and kernels all work, independent of the user's own query:
```bash
# fastest, fully offline (~30s, verified): single mode combo, no MSA server call
pixi run -e <env> pytest openfold3/tests/inference/test_inference_full.py \
    -k "ubiquitin and no_msa and no_templates" -v --log-cli-level=INFO

# broader (~2min, verified): all 4 MSA-server/template combos for this one case,
# including a real ColabFold server call for the msa-server combos
pixi run -e <env> pytest openfold3/tests/inference/test_inference_full.py -k ubiquitin -v --log-cli-level=INFO
```
Requires an accelerator (CUDA/ROCm/MPS) and downloaded model weights; skips cleanly otherwise. Drop `-k ubiquitin` entirely to run the full parametrized suite (every bundled case × every combo) — much slower, only worth it after a real config change, not as a pre-flight check.

**b. Run the user's actual query.json once, with defaults, into a scratch dir, and inspect the full output tree** — not just the aggregated confidences. Confirmed by an actual run (`query_ubiquitin.json`, `--use-msa-server false`, 1 seed, 1 sample):
```
<output_dir>/
  summary.txt                    # Total/Successful/Failed query counts — check Failed == 0 first
  experiment_config.json         # resolved settings actually used (catches "my runner.yml override didn't apply")
  model_config.json
  inference_query_set.json       # your query.json after validation/normalization — catches silent misparses
  msas/...                       # populated even with --use-msa-server false (dummy single-seq MSA)
  <query_key>/seed_<n>/
    <query_key>_seed_<n>_sample_<i>_model.cif
    <query_key>_seed_<n>_sample_<i>_confidences.json
    <query_key>_seed_<n>_sample_<i>_confidences_aggregated.json
    timing.json
```
Things worth actually looking at, not just confirming they exist:
- `summary.txt`: `Failed Queries` should be 0.
- `inference_query_set.json`: the chains/molecule types OF3 actually parsed — the fastest way to catch a query.json that validated but didn't mean what you thought (wrong chain merged, sequence typo, ligand not recognized).
- `*_confidences_aggregated.json`: `has_clash` should be 0.0 and `avg_plddt` in a plausible range (not near 0, which usually means something upstream was wrong, not that the fold is just bad).
- Open the `.cif` in a viewer if this is a query you care about getting right, not just a mechanical check.

Only after both checks look right should you scale up to real `--num-model-seeds`/`--num-diffusion-samples`, the full query batch, or a non-default kernel.

## 3. Common configuration modes

### MSA source (mutually exclusive; pick one)

| Mode | Flag(s) | Notes |
|---|---|---|
| ColabFold MSA server (default) | `--use-msa-server true` | Only protein sequences are sent to the server (not the raw query — species/sequence info for alignment). Best for a handful of ad hoc predictions. |
| No MSA server (single-sequence for chains without other MSA info) | `--use-msa-server false` | Prediction quality is generally worse without MSAs. Combine with precomputed MSAs below if you have them, otherwise it's effectively MSA-free. |
| Precomputed MSAs | `--use-msa-server false` + `main_msa_file_paths`/`paired_msa_file_paths` set per-chain in `query.json` | For high-throughput/repeated screening. Full directory layout and pairing rules: see `docs/source/precomputed_msa_how_to.md`. Don't hand-roll this from scratch — read that doc first. |
| Precompute alignments separately, then predict | `run_openfold align-msa-server --query-json queries.json --output-dir output/alignments` first (writes `query_msa.json`), then `predict --query-json output/alignments/query_msa.json --use-msa-server false` | Useful to batch/dedupe server calls across many queries before running the (possibly repeated/expensive) model forward pass. |

### Templates

```bash
--use-templates true   # off by default
```
Requires either ColabFold-derived template alignments (automatic when `--use-msa-server true --use-templates true`), precomputed `template_alignment_file_path`, or `template_cif_paths`/`template_cif_chain_ids` set directly in the query (CIF-direct mode, no alignment step needed — see `docs/source/template_how_to.md`).

### Seeds and diffusion samples

```bash
--num-model-seeds 4 --num-diffusion-samples 5
```
Produces `num_queries × num_model_seeds` forward passes, each yielding `num_diffusion_samples` structures. For an explicit seed list (not just a count), set `experiment_settings.seeds: [100, 101, ...]` in a runner.yml instead.

### Output format / content — via `runner.yml`

```yaml
output_writer_settings:
  structure_format: pdb          # default: cif
  write_features: True           # save input features
  write_latent_outputs: True     # save si_trunk/zij_trunk/atom_positions_predicted as *_latent_output.pt
  write_full_confidence_scores: False   # skip per-atom confidences to save disk on large seed/sample sweeps
```
Bundled starting point: `examples/example_runner_yamls/output_settings.yml`.

### Low-memory mode (large complexes / limited VRAM)

```yaml
model_update:
  presets:
    - predict   # required alongside low_mem
    - low_mem
```
Trades speed for memory (pairformer output computed sequentially across diffusion samples). Bundled: `examples/example_runner_yamls/low_mem.yml`.

### Multi-GPU / multi-node

```yaml
pl_trainer_args:
  devices: 4       # default: 1
  num_nodes: 1     # default: 1
```
Bundled: `examples/example_runner_yamls/multiple_gpu.yml`.

### Evoformer kernel choice (cuequivariance / triton)

Swap in one of the example runner yamls under `examples/example_runner_yamls/` — `cuequivariance.yml` or `triton.yml` — rather than hand-writing the `model_update` block, e.g.:
```bash
--runner-yaml examples/example_runner_yamls/cuequivariance.yml
```
Note some kernels need a matching pixi env (cuequivariance needs `openfold3-cuda12-pypi`, not `openfold3-cuda12`; triton needs a ROCm or CUDA env with Triton installed). Deepspeed's evoformer attention kernel (`use_deepspeed_evo_attention`) still exists but is no longer a recommended default — prefer cuequivariance or triton. For systematically comparing kernels across queries/lengths rather than a single run, use the `of3-run-examples` skill instead.

### Pocket constraints

Only relevant if the `query.json` has a `pocket_constraint` block (see `of3-build-query`). Enabled by default; to disable without editing the query:
```yaml
dataset_config_kwargs:
  pocket_sampling:
    enabled: False
```
(`docs/source/inference.md` has a typo here — `datset_config_kwargs` — that silently no-ops if copied verbatim; the real field, confirmed in `openfold3/entry_points/validator.py`, is `dataset_config_kwargs`.)

## 4. Combining multiple settings

Runner-yaml keys merge via `config_utils.deep_update`, so a single YAML can combine several of the blocks above (e.g. low_mem + pdb output + 4 GPUs). CLI flags always win over whatever the YAML says for the flags that exist (`--use-msa-server`, `--use-templates`, `--num-model-seeds`, `--num-diffusion-samples`, `--output-dir`, `--inference-ckpt-path`/`--inference-ckpt-name`).

For settings with no CLI flag, write a scratch runner.yml (e.g. in the scratchpad dir) and pass it via `--runner-yaml`. Check `examples/reference_full_config/full_config.yml` for the full list of overridable keys before inventing a new one.

## 5. Output layout

```
<output_dir>/<query_key>/seed_<n>/
  <query_key>_seed_<n>_sample_<i>_model.cif   # or .pdb
  <query_key>_seed_<n>_sample_<i>_confidences.json          # plddt, pae, pde
  <query_key>_seed_<n>_sample_<i>_confidences_aggregated.json  # ptm, iptm, sample_ranking_score, ...
  timing.json                                  # runtime_s — authoritative runtime, excludes MSA/load overhead
```
Use `sample_ranking_score` in the aggregated confidences file to rank samples within a seed. MSAs (if server-generated) are cached under `<output_dir>/msas/<run-id>/`. Full field reference: `docs/source/inference.md` §4.