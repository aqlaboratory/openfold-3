# PDB training subset scripts

Tools for pulling a small, working slice of the PDB training set from S3 and
running a quick end-to-end training smoke test against it -- without having
to download the full ~1.7GB training cache's worth of structures.

## Files

| File | Purpose |
| --- | --- |
| [`pdb_subset_helpers.py`](pdb_subset_helpers.py) | All the logic: cache sampling, S3 download, runner yaml generation. Import target, not run directly. |
| [`download_subset.py`](download_subset.py) | CLI entry point. Creates the subset cache(s) and downloads the files they reference. |
| [`train_on_subset.sh`](train_on_subset.sh) | One-shot integration test: runs `download_subset.py`, then `run_openfold train`. |
| `train_pdb_subset.yaml` | Generated (not committed) -- the `run_openfold train` runner config, written by `download_subset.py` each time it runs. |
| `training_cache_with_templates_subset_{N}.json`, `validation_cache_with_templates_subset_{N}.json` | Generated (not committed) -- the sampled dataset caches. |
| `pdb_training_set/` | Generated (not committed) -- the downloaded structures/alignments/templates/reference mols, laid out to match `run_openfold`'s expected directory structure. |

The generated files above are git-ignored by convention (large, host-specific
absolute paths); only the three source files are meant to be committed.

## Step 1: Download the full dataset cache

The full, un-sampled caches live on S3 at:

```
s3://openfold3-data/pdb_training_set/dataset_caches/training_cache_with_templates.json   (~1.7GB, 180975 structures)
s3://openfold3-data/pdb_training_set/dataset_caches/validation_cache_with_templates.json  (~36MB, 1719 structures)
```

You don't normally need to do anything for this step -- `download_subset.py`
downloads whichever one it needs automatically, the first time it's asked to
sample a subset and doesn't find a local copy. It also sanitizes the file:
these caches contain bare `NaN` literals (e.g. `"resolution": NaN` for NMR
structures with no resolution), which is invalid JSON and would otherwise
crash the streaming parser used to sample from it.

**Where this lives:** `download_full_cache()` and `FULL_CACHE_S3_KEYS` in
`pdb_subset_helpers.py`. If the S3 layout of the full caches ever changes,
update `FULL_CACHE_S3_KEYS`. If you already have a full cache downloaded
elsewhere, point `--train-cache`/`--val-cache` (see Step 2) at it instead of
re-downloading.

## Step 2: Create a sample training set

```bash
pixi run --manifest-path ../../pixi.toml -e openfold3-cuda12 python download_subset.py
```

This does three things, in order:

1. **Sample a subset cache**, if one for the requested size doesn't already
   exist: `--train-size` (default 32) structures are randomly drawn from the
   full training cache, `--val-size` (default 16) from the full validation
   cache, both with a fixed `--seed` (default 42) for reproducibility.
   Writes `training_cache_with_templates_subset_{N}.json` /
   `validation_cache_with_templates_subset_{N}.json` next to the script.
   **Where this lives:** `sample_subset_cache()` / `get_or_create_subset_cache()`.

2. **Download only the files that subset actually references** -- target
   structures, alignment arrays, and template caches/structure arrays, into
   `--output-dir` (default `pdb_training_set/`, matching what `run_openfold`
   expects). Reference-molecule SDFs are downloaded in a second pass, using
   the *actual* residue composition of the downloaded structures (not just
   the cache metadata) -- this matters because non-standard/modified
   residues embedded in a protein/RNA/DNA chain (e.g. a modified cysteine
   like `CME`) have no `reference_mol_id` in the cache and can only be
   discovered by inspecting the downloaded structure's `res_name` field.
   Skipping the full `reference_mols/` sync (~68k files) this way is the
   main reason this script exists instead of `aws s3 sync`.
   **Where this lives:** `build_structure_manifest()`,
   `build_reference_mol_manifest()`, `scan_structure_residue_names()`,
   `download_manifest()`.

3. **Write a runner yaml** (`--runner-yaml`, default
   `train_pdb_subset.yaml` next to the script) with `dataset_paths` pointing
   at whatever `--output-dir`/cache files were actually produced. This is
   what makes the yaml self-consistent with however the script was invoked
   -- it's regenerated every run, not hand-maintained.
   **Where this lives:** `build_runner_yaml_config()` / `write_runner_yaml()`.
   Everything *except* `dataset_paths` (model config, crop settings, trainer
   args, etc.) is fixed boilerplate baked into `build_runner_yaml_config()`
   -- edit that function if you need to change training hyperparameters for
   this workflow.

Useful flags:

```bash
python download_subset.py --verify                # check completeness, don't download
python download_subset.py --train-size 128 --val-size 32
python download_subset.py --output-dir /data/foo
python download_subset.py --train-cache /path/to/training_cache_with_templates.json
```

`--verify` can only fully check reference-mol completeness if the target
structures are already downloaded (it needs to read them to know which
residues -- and therefore which reference mols -- are required); otherwise
it prints a note and only verifies structures/alignments/templates.

## Step 3: Run the training integration test

```bash
./train_on_subset.sh
```

This just chains steps 1-2 (via `download_subset.py`, idempotent -- already
-downloaded files are skipped) with `run_openfold train
--runner-yaml=train_pdb_subset.yaml`. Any extra arguments are passed through
to `run_openfold train`, e.g.:

```bash
./train_on_subset.sh --seed 1234
```

**Where this lives:** `train_on_subset.sh`. If you need a different subset
size or output location for the integration test specifically, either edit
the `download_subset.py` invocation in this script to pass the relevant
flags, or run `download_subset.py` yourself first with the flags you want
and then invoke `run_openfold train --runner-yaml=...` directly.

## Modifying things

- **Change what's sampled/downloaded:** `pdb_subset_helpers.py`. The three
  sections (subset cache creation / S3 download / runner yaml generation)
  are marked with `# ---` banners.
- **Change CLI flags or the overall download flow:** `download_subset.py`'s
  `main()`.
- **Change training hyperparameters used by the integration test:**
  `build_runner_yaml_config()` in `pdb_subset_helpers.py` (everything under
  `dataset_paths` is derived automatically -- don't hand-edit those keys, or
  your edits will be overwritten the next time `download_subset.py` runs).
- **Change what counts as a "safety net" reference mol** (always downloaded
  regardless of what's scanned from structures): `AMINO_ACID_CCD_CODES` /
  `STANDARD_NUCLEOTIDE_CCD_CODES` in `pdb_subset_helpers.py`.
