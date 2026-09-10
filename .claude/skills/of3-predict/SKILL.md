---
name: of3-predict
description: "Run an OpenFold3 (OF3) structure prediction on a query.json via `run_openfold predict`, with guidance on common configuration toggles: ColabFold MSA server on/off, precomputed MSAs, templates on/off, seeds/diffusion samples, output format (cif/pdb), low-memory preset, multi-GPU, evoformer kernel choice (cuequivariance/triton), and pocket constraints. Use when asked to run/predict/fold a structure from a query.json, or to change how an OF3 prediction is configured. For building the query.json itself, use of3-build-query first."
---

# Running an OF3 prediction

## Read the inference docs first

`docs/source/inference.md` is the authoritative guide to running and configuring inference — **read the relevant section before changing any setting** rather than working from memory. This skill does not restate it.

| What you need | Where |
|---|---|
| MSA modes (ColabFold server / precomputed / MSA-free), incl. `align-msa-server` pre-batching | `inference.md` §3.2, §3.4 |
| `runner.yml` mechanics, persistent `$OPENFOLD_CACHE/runner.yml`, model_update customization | `inference.md` §3.3 |
| Seeds & diffusion samples (`--num-model-seeds`, `experiment_settings.seeds`) | `inference.md` §3.3 🌱 |
| Output format & content (`structure_format: pdb`, `write_latent_outputs`, `write_full_confidence_scores`) | `inference.md` §3.3 📦/⏩, §4.5 |
| Low-memory mode, multi-GPU/multi-node, AMD ROCm/Triton, Apple Silicon MPS | `inference.md` §3.3 🧠/🖥️/🔴/🍎 |
| Templates, incl. CIF-direct mode | `inference.md` §3.3 🧬, `docs/source/template_how_to.md` |
| Precomputed MSA layout and pairing rules | `docs/source/precomputed_msa_how_to.md` |
| Pocket constraints (enable/disable, sampling settings) | `inference.md` §3.5 |
| Kernel choice + which pixi env each kernel needs | `docs/source/kernels.md` |
| Output directory layout and every confidence field | `inference.md` §4 (§4.1 predictions, §4.2–4.3 MSAs, §4.4 metadata) |
| Full list of overridable config keys | `examples/reference_full_config/full_config.yml`, `docs/source/configuration_reference.md` |

Ready-made runner yamls for the common cases (`low_mem.yml`, `multiple_gpu.yml`, `output_settings.yml`, `cuequivariance.yml`, `triton.yml`) are in `examples/example_runner_yamls/` — pass one with `--runner-yaml` instead of hand-writing the `model_update` block.

## Minimal command

```bash
run_openfold predict \
    --query-json /path/to/query.json \
    --output-dir /path/to/output/
```

Defaults: ColabFold MSA server on, no templates, 1 seed, 5 diffusion samples, `.cif` output.

Configuration comes from three layers, later overriding earlier: `$OPENFOLD_CACHE/runner.yml` (default `~/.openfold3/runner.yml`, applied automatically if present) → `--runner-yaml <path>` → CLI flags. Runner-yaml keys merge via `config_utils.deep_update`, so one YAML can combine several blocks (e.g. low_mem + pdb output + 4 GPUs). CLI flags always win for the flags that exist (`--use-msa-server`, `--use-templates`, `--num-model-seeds`, `--num-diffusion-samples`, `--output-dir`, `--inference-ckpt-path`/`--inference-ckpt-name`); for settings with no flag, write a scratch runner.yml and pass it via `--runner-yaml`.

## Prerequisites

- A `query.json` (see the `of3-build-query` skill if it doesn't exist yet). Just smoke-testing? `examples/example_inference_inputs/` has ready-to-run samples — `query_ubiquitin.json` is the smallest.
- An activated OF3 environment. If unsure which pixi environment to use, run `pixi environment list` (or check `pixi.toml`) in the repo root — GPU users on CUDA typically use `openfold3-cuda12`; CPU/macOS uses `openfold3-base`. Prefix commands with `pixi run -e <env>` if not already inside an activated shell. Note the kernel presets need matching envs (see `kernels.md`): cuequivariance needs `openfold3-cuda12-pypi`/`openfold3-cuda13-pypi`, not plain `openfold3-cuda12`; triton needs a ROCm or CUDA env with Triton installed.
- Model checkpoint: auto-downloaded to `$OPENFOLD_CACHE` (default `~/.openfold3/`) on first run if `--inference-ckpt-path`/`--inference-ckpt-name` aren't given.

## Validate before a real run

Before committing to the user's actual (possibly large/expensive) job, confirm the environment and the query are both sound. Two independent checks — offer both, they catch different things:

**a. Environment/install smoke test** — runs a small built-in case end-to-end and checks the numeric output against known-good ranges. Confirms the accelerator, checkpoint, and kernels all work, independent of the user's own query:
```bash
# fastest, fully offline (~30s, verified): single mode combo, no MSA server call
pixi run -e <env> pytest openfold3/tests/inference/test_inference_full.py \
    -k "ubiquitin and no_msa and no_templates" -v --log-cli-level=INFO

# broader (~2min, verified): all 4 MSA-server/template combos for this one case,
# including a real ColabFold server call for the msa-server combos
pixi run -e <env> pytest openfold3/tests/inference/test_inference_full.py -k ubiquitin -v --log-cli-level=INFO
```
Requires an accelerator (CUDA/ROCm/MPS) and downloaded model weights; skips cleanly otherwise. Drop `-k ubiquitin` to run the full parametrized suite (every bundled case × every combo) — much slower, only worth it after a real config change, not as a pre-flight check.

**b. Run the user's actual query.json once, with defaults, into a scratch dir, and inspect the output tree.** Layout is documented in `inference.md` §4; what's worth actually *looking at* rather than confirming exists:
- `summary.txt` (top level, not in the docs): `Failed Queries` should be 0 — check this first.
- `inference_query_set.json`: the chains/molecule types OF3 actually parsed — the fastest way to catch a query.json that validated but didn't mean what you thought (wrong chain merged, sequence typo, ligand not recognized).
- `experiment_config.json`: the resolved settings actually used — catches "my runner.yml override didn't apply".
- `*_confidences_aggregated.json`: `has_clash` should be 0.0 and `avg_plddt` in a plausible range (near 0 usually means something upstream was wrong, not that the fold is just bad). Rank samples within a seed by `sample_ranking_score`.
- Open the `.cif` in a viewer if this is a query you care about getting right, not just a mechanical check.

Note `msas/` is populated even with `--use-msa-server false` (dummy single-sequence MSA), so its presence is not evidence the server ran.

Only after both checks look right should you scale up to real `--num-model-seeds`/`--num-diffusion-samples`, the full query batch, or a non-default kernel.

## Gotchas

- Pocket sampling only matters if the `query.json` actually has a `pocket_constraint` block (see `of3-build-query`); it's enabled by default when one is present, and disabled via `dataset_config_kwargs.pocket_sampling.enabled` (`inference.md` §3.5).
- Deepspeed's evoformer attention kernel (`use_deepspeed_evo_attention`) still exists but is no longer a recommended default — prefer cuequivariance or triton.
- `timing.json`'s `runtime_s` is the authoritative per-query runtime; it excludes MSA computation and model loading, so wall-clock will be longer.
