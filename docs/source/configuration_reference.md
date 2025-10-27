# OpenFold3 Configuration Reference

The [`full_config.yml` file](https://github.com/aqlaboratory/openfold-3/blob/feature/update-docs/examples/reference_full_config/full_config.yml) is a comprehensive reference configuration file that demonstrates all available configuration options for OpenFold3 inference and training experiments. This file is located at `examples/reference_full_config/full_config.yml` and serves as a complete example of all configurable settings.

## Overview

The configuration file is organized into several main sections. Each section corresponds to a specific Pydantic model class defined in the OpenFold3 codebase. When you provide a `runner.yml` file, it **overrides** the default settings defined in [`validator.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py).

## Important Notes

- **Selective Configuration**: Only specify the settings you want to override in your runner YAML file. All unspecified options will use their default values.
- **Command-line Priority**: Command-line arguments take precedence and will override any values specified in the YAML file.
- **Reference Implementation**: The full configuration file serves as a reference - create your own simplified runner YAML based on your specific needs. See `examples/example_runner_yamls/` for common usage examples.

## Configuration Sections

### 1. Experiment Settings (`experiment_settings`)

Defines overall experiment parameters, including execution mode and seed configuration.

**Pydantic Model**: [`InferenceExperimentSettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L247)

**All Options**:
- `mode` *(ValidModeType)*: Experiment mode - `predict` or `train` (default: `predict`)
- `output_dir` *(Path)*: Directory where outputs will be written (default: `./`)
- `log_dir` *(Path | None)*: Directory for logs (default: `null`)
- `seeds` *(int | list[int])*: Starting seed or list of random seeds for inference (default: `[42]`)
- `num_seeds` *(int | None)*: Number of seeds to generate if only a starting seed is provided (default: `null`)
- `use_msa_server` *(bool)*: Whether to use ColabFold MSA server (default: `false`)
- `use_templates` *(bool)*: Whether to use template structures (default: `false`)
- `skip_existing` *(bool)*: Skip results that already exist (default: `false`)

**Example**:
```yaml
experiment_settings:
  mode: predict
  output_dir: ./results
  seeds: [42, 100, 200]
  use_msa_server: true
```

---

### 2. PyTorch Lightning Trainer Args (`pl_trainer_args`)

Configures the PyTorch Lightning trainer for distributed training and multi-GPU inference.

**Pydantic Model**: [`PlTrainerArgs`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L121)

**All Options**:
- `max_epochs` *(int)*: Maximum number of training epochs (default: `1000`)
- `accelerator` *(str)*: Device type - `gpu` or `cpu` (default: `gpu`)
- `precision` *(int | str)*: Numerical precision - `32-true`, `16-mixed`, etc. (default: `32-true`)
- `num_nodes` *(int)*: Number of compute nodes (default: `1`)
- `devices` *(int)*: Number of GPUs per node (default: `1`)
- `profiler` *(str | None)*: Profiler to use (default: `null`)
- `log_every_n_steps` *(int)*: Logging frequency in steps (default: `1`)
- `enable_checkpointing` *(bool)*: Enable checkpointing (default: `true`)
- `enable_model_summary` *(bool)*: Enable model summary (default: `false`)
- `deepspeed_config_path` *(Path | None)*: Path to DeepSpeed configuration file (default: `null`)
- `distributed_timeout` *(timedelta | None)*: Timeout for distributed operations (default: `PT30M`)
- `mpi_plugin` *(bool)*: Use MPI plugin (default: `false`)

**Example**:
```yaml
pl_trainer_args:
  devices: 4
  num_nodes: 1
  precision: 16-mixed
```

---

### 3. Model Update (`model_update`)

Specifies model presets and custom architecture modifications.

**Pydantic Model**: [`ModelUpdate`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/project_entry.py#L28)

**All Options**:
- `presets` *(list[str])*: List of model presets to apply (default: `[]`)
  - `predict`: Inference configuration (required for inference)
  - `low_mem`: Low memory mode for large structures
  - `pae_enabled`: Enable Predicted Aligned Error (PAE) head
- `custom` *(dict)*: Custom model configuration overrides (default: `{}`)

**Example**:
```yaml
model_update:
  presets:
    - predict
    - pae_enabled
    - low_mem
  custom: {}
```

---

### 4. Checkpoint and Cache Paths

**Pydantic Model**: Fields on [`InferenceExperimentConfig`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L347)

**All Options**:
- `inference_ckpt_path` *(Path | None)*: Path to model checkpoint file (`.pt` file)
  - Default: `$HOME/.openfold3/of3_ft3_v1.pt`
  - Will download parameters if not present
- `cache_path` *(Path | None)*: Directory for storing cached model parameters
  - Default: `$HOME/.openfold3/`

---

### 5. Data Module Args (`data_module_args`)

Configures data loading and processing.

**Pydantic Model**: [`DataModuleArgs`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L110)

**All Options**:
- `batch_size` *(int)*: Batch size (default: `1`)
- `data_seed` *(int | None)*: Random seed for data processing (default: `42`)
- `num_workers` *(int)*: Number of data loading workers (default: `10`)
- `num_workers_validation` *(int)*: Number of workers for validation (default: `4`)
- `epoch_len` *(int)*: Length of training epoch (default: `4`)

**Example**:
```yaml
data_module_args:
  batch_size: 1
  num_workers: 8
```

---

### 6. Dataset Config Kwargs (`dataset_config_kwargs`)

Configures MSA and template feature generation.

**Pydantic Model**: [`InferenceDatasetConfigKwargs`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/dataset_configs.py#L270)

**All Options**:
- `ccd_file_path` *(FilePath | None)*: Path to Chemical Component Dictionary file, uses CCD from Biotite if null (default: `null`)
- `msa` *(MSASettings)*: MSA processing settings (see below)
- `template` *(TemplateSettings)*: Template processing settings (see below)

#### MSA Settings (`msa`)

Controls how MSAs are parsed and processed into features.

**Pydantic Model**: [`MSASettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/dataset_config_components.py#L32)

**All Options**:
- `max_rows_paired` *(int)*: Maximum rows for paired MSAs (default: `8191`)
- `max_rows` *(int)*: Maximum total MSA rows (default: `16384`)
- `subsample_with_bands` *(bool)*: Use MMSeqs2-style subsampling (default: `false`, not currently supported)
- `min_chains_paired_partial` *(int)*: Minimum chains for partial pairing (default: `2`)
- `pairing_mask_keys` *(list[str])*: Masks to apply during pairing (default: `["shared_by_two", "less_than_600"]`)
- `moltypes` *(list[MoleculeType])*: Molecule types to process (default: `[0, 1]` for protein and RNA)
- `max_seq_counts` *(dict)*: Max sequences per MSA file (default includes: uniref90_hits: 10000, uniprot_hits: 50000, etc.)
- `msas_to_pair` *(list[str])*: MSA files to use for online pairing (default: `["uniprot_hits", "uniprot"]`)
- `aln_order` *(list)*: Order to vertically concatenate MSA files (default includes: uniref90_hits, bfd_uniclust_hits, etc.)
- `paired_msa_order` *(list)*: Order to vertically concatenrate pre-paired MSAs (default: `["colabfold_paired"]`)

**Example**:
```yaml
dataset_config_kwargs:
  msa:
    max_rows: 16384
    max_rows_paired: 8191
    moltypes: [0, 1]  # protein and RNA
```

#### Template Settings (`template`)

Controls template structure processing.

**Pydantic Model**: [`TemplateSettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/dataset_config_components.py#L113)

**All Options**:
- `n_templates` *(int)*: Number of templates to use (default: `4`)
- `take_top_k` *(bool)*: Use top K templates by quality (default: `false`)
- `distogram` *(TemplateDistogramSettings)*: Distogram binning settings
  - `min_bin` *(float)*: Minimum distance bin (default: `3.25`)
  - `max_bin` *(float)*: Maximum distance bin (default: `50.75`)
  - `n_bins` *(int)*: Number of bins (default: `39`)

**Example**:
```yaml
dataset_config_kwargs:
  template:
    n_templates: 4
    take_top_k: true
```

---

### 7. Output Writer Settings (`output_writer_settings`)

Configures the format of output files.

**Pydantic Model**: [`OutputWritingSettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L141)

**All Options**:
- `structure_format` *(Literal["pdb", "cif"])*: Output format (default: `cif`)
- `full_confidence_output_format` *(Literal["json", "npz"])*: Confidence output format (default: `json`)
- `write_features` *(bool)*: Write intermediate features (default: `false`)
- `write_latent_outputs` *(bool)*: Write model intermediate outputs (default: `false`)

**Example**:
```yaml
output_writer_settings:
  structure_format: pdb
  full_confidence_output_format: json
```

---

### 8. MSA Computation Settings (`msa_computation_settings`)

Configures the ColabFold MSA server integration.

**Pydantic Model**: [`MsaComputationSettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/data/tools/colabfold_msa_server.py#L904)

**All Options**:
- `msa_file_format` *(Literal["npz", "a3m"])*: Format for saved MSAs (default: `npz`)
- `server_user_agent` *(str)*: User agent string (default: `openfold`)
- `server_url` *(Url)*: ColabFold server URL (default: `https://api.colabfold.com`)
- `save_mappings` *(bool)*: Save sequence ID mappings (default: `true`)
- `msa_output_directory` *(Path)*: Directory for MSA outputs (default: temporary directory)
- `cleanup_msa_dir` *(bool)*: Delete MSAs after processing (default: `true`)

**Example**:
```yaml
msa_computation_settings:
  msa_file_format: npz
  cleanup_msa_dir: false
  msa_output_directory: /path/to/msas
```

---

### 9. Template Preprocessor Settings (`template_preprocessor_settings`)

Configures template structure preprocessing and filtering.

**Pydantic Model**: [`TemplatePreprocessorSettings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/data/pipelines/preprocessing/template.py#L1459)

**All Options**:
- `mode` *(Literal["train", "predict"])*: Processing mode (default: `predict`)
- `moltypes` *(list[MoleculeType])*: Molecule types to process (default: `[0]` for protein)
- `max_sequences_parse` *(int)*: Max sequences to parse (default: `200`)
- `max_seq_id` *(float | None)*: Maximum sequence identity threshold (default: `null`)
- `min_align` *(float | None)*: Minimum alignment coverage (default: `null`)
- `min_len` *(int | None)*: Minimum aligned residues (default: `null`)
- `max_release_date` *(datetime | None)*: Maximum template release date (default: `null`)
- `min_release_date_diff` *(int | None)*: Minimum days between query and template release (default: `null`)
- `max_templates` *(int)*: Maximum templates per chain (default: `20`)
- `fetch_missing_structures` *(bool)*: Fetch missing structures from PDB (default: `true`)
- `create_precache` *(bool)*: Cache template structure data (default: `false`)
- `preparse_structures` *(bool)*: Preparse structures into .npz files (default: `false`)
- `create_logs` *(bool)*: Create preprocessing logs (default: `false`)
- `n_processes` *(int)*: Number of preprocessing processes (default: `1`)
- `chunksize` *(int)*: Tasks per worker in multiprocessing (default: `1`)
- `structure_directory` *(Path | None)*: Directory for template structures (default: `null`)
- `structure_file_format` *(str)*: File format of structures - `cif` or `pdb` (default: `cif`)
- `output_directory` *(Path | None)*: Output directory for templates (default: `null`)
- `precache_directory` *(Path | None)*: Directory for template precache (default: `null`)
- `structure_array_directory` *(Path | None)*: Directory for preparsed structures (default: `null`)
- `cache_directory` *(Path | None)*: Directory for template cache (default: `null`)
- `log_directory` *(Path | None)*: Directory for logs (default: `null`)
- `ccd_file_path` *(Path | None)*: Path to Chemical Component Dictionary file (default: `null`)

**Example**:
```yaml
template_preprocessor_settings:
  mode: predict
  max_templates: 20
  fetch_missing_structures: true
```

---

## Related Documentation

- [`Inference Guide`](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/Inference.md) - How to run inference
- [`Input Format`](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/input_format.md) - Query JSON format
- [`Precomputed MSA Guide`](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/precomputed_msa_how_to.md) - Using precomputed MSAs
- [`Precomputed MSA Generation Guide`](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/precomputed_msa_generation_how_to.md) - Generating precomputed MSAs
- [`Template Guide`](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/template_how_to.md) - Template processing guide

## Default Values Reference

For the complete list of default values, see the Pydantic model classes in:
- [`openfold3/entry_points/validator.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py) - Main configuration classes
- [`openfold3/projects/of3_all_atom/config/dataset_config_components.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/dataset_config_components.py) - MSA and template settings
- [`openfold3/core/data/tools/colabfold_msa_server.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/data/tools/colabfold_msa_server.py) - MSA server settings
- [`openfold3/core/data/pipelines/preprocessing/template.py`](http://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/data/pipelines/preprocessing/template.py) - Template preprocessing settings

