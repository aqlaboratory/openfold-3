# OpenFold3 Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 implements the features described in [AlphaFold 3 *Nature* paper](https://www.nature.com/articles/s41586-024-07487-w).

This guide covers how to use OpenFold3 to make structure predictions.


## 1. Inference features

OpenFold3 replicates the full set of input features described in the *AlphaFold 3* publication. All features of AlphaFold 3 are **fully implemented and supported in training**. We are actively working on integrating the same functionalities into the inference pipeline. 

Below is the current status of inference feature support by molecule type:


### 1.1 Protein

Supported:

- Prediction with MSA
    - using ColabFold MSA pipeline
    - using pre-computed MSAs
- Prediction without MSA
- OpenFold3's own MSA generation pipeline
- Template-based prediction
    - using ColabFold template alignments
    - using pre-computed template alignments
- Non-canonical residues

Coming soon:

- Covalently modified residues and other cross-chain covalent bonds
- User-specified template structures (as opposed to top 4)

### 1.2 DNA

Supported:

- Prediction without MSA (per AF3 default)
- Non-canonical residues

Coming soon:

- Covalently modified residues and other cross-chain covalent bonds


### 1.3 RNA

Supported:

- Prediction with MSA, using OpenFold3's own MSA generation pipeline
- Prediction without MSA
- OpenFold3's own MSA generation pipeline
- Non-canonical residues

Coming soon:

- Template-based prediction
- Covalently modified residues and other cross-chain covalent bonds
- Protein-RNA MSA pairing


### 1.4 Ligand

Supported:

- Non-covalent ligands

Coming soon:

- Covalently bound ligands
- Polymeric ligands such as glycans


## 2. Pre-requisites:

- OpenFold3 Conda Environment. See {ref}`OpenFold3 Installation <openfold3-installation>` for instructions on how to build this environment.
- OpenFold3 Model Parameters. See {ref}`Openfold3 Parameters <openfold3-parameters>` for how to download these parameters.


## 3. Running OpenFold3 Inference

A prediction job can be submitted with the following command:

```bash
run_openfold --query_json=<query_json>
```

Sample input query jsons can be found in the [examples/example_inference_inputs](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs) directory.

Full output directories are provided on the [OpenFold HuggingFace repo](https://huggingface.co/OpenFold/OpenFold3/tree/main/examples). These include:
- Single-chain protein (monomer) -- Ubiquitin (PDB: 1UBQ)
- Multi-chain protein with identical chains (homomer) -- GCN4 leucine zipper (PDB: 2ZTA)
- Multi-chain protein with different chains (multimer) -- Deoxy human hemoglobin (PDB: 1A3N)
- Protein-ligand complex -- Mcl-1 with small molecule inhibitor (PDB: 5FDR)


### 3.1 Input Data: Query JSON

Queries can include any combination of single- or multi-chain proteins, with or without ligands, and may contain multiple such complexes. <br/>
Input is provided via a `query.json` file — a structured JSON document that defines each query, its constituent chains, chain types (e.g., protein, DNA, ligand) and sequences or molecular graphs. Optionally, the query can include paths to precomputed protein or RNA MSAs. <br/>
See {doc}`OpenFold3 input format <input_format>` for instructions on how to specify your input data.


### 3.2 Inference Modes by MSA Input
OpenFold3 currently supports three inference modes with respect to MSAs:

- 🚀 With ColabFold MSA Server (default)
- 📂 With Precomputed MSAs
- 🚫 Without MSAs (MSA-free)

Each mode shares the same command structure but differs in how MSAs are provided or generated.

#### 3.2.1 🚀 Inference with ColabFold MSA Server (Default)

This mode automatically generates MSAs using the ColabFold server. Only protein sequences are sent to the server. We recommend this mode if you only have a couple of structures to predict.

```bash
run_openfold predict \
    --query_json /path/to/query.json \
    --inference_ckpt_path /path/to/inference.ckpt \
    --use_msa_server \
    --output_dir /path/to/output/
```

This command uses the `run_openfold` binary, for which the source code is available [here](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/run_openfold.py).

**Required arguments**

- `--query_json` *(Path)*
    - Path to the input query JSON file.

**Optional arguments**

- `--inference_ckpt_path` *(Path)*
    - Path to a model checkpoint file (`.pt` file)
    - Will use a default checkpoint if not specified, which will be downloaded on the first inference run. See {ref}`OpenFold Parameters <openfold3-parameters>` for information

- `--use_msa_server` *(bool, optional, default = True)*
    - Whether to use the ColabFold server for MSA generation.

- `--output_dir` *(Path, optional, default = `test_train_output/`)*
    - Directory where outputs will be written.

- `--num_diffusion_samples` *(int, optional, default = 5)*
    - Number of diffusion samples per query.

- `--num_model_seeds` *(int, optional, default = 1)*
    - Number of random seeds to use per query.
    - To manually select specific seeds, please use the `runner.yml` and refer to the {ref}`Custom Random Seeds section <custom-random-seeds-inference>` below.

- `--runner_yaml` *(Path, optional, default = null)*
    - YAML config for full control over model and data parameters. See the {doc}`configuration reference <configuration_reference>` and [full configuration reference file](https://github.com/aqlaboratory/openfold-3/blob/main/examples/reference_full_config/full_config.yml) for all available options.
    - See the {ref}`runner yaml section below for more information <33-customized-inference-settings-using-runneryml>` 

📝  *Notes*: 
- Only protein sequences are submitted to the ColabFold server so this mode only uses MSAs for protein chains.
- All arguments can also be set via `runner_yaml`, but command-line flags take precedence and will override values specified in the YAML file (see [Customized Inference Settings](33-customized-inference-settings-using-runneryml) for details).


#### 3.2.2 📂 Inference with Precomputed MSAs
This mode allows inference using MSA files prepared manually or by external tools. We recommend this mode for high-throughput screeing applications where you want to run hundreds or thousands of predictions. See the {doc}`precomputed MSA documentation <precomputed_msa_how_to>` for a step-by-step tutorial, the {doc}`MSA generation guide <precomputed_msa_generation_how_to>` for using our MSA generation pipeline and the {doc}`precomputed MSA explanatory document <precomputed_msa_explanation>` for a more in-depth explanation on how precomputed MSA handling works. 

An example query json with a sample of how the alignment directories can be formatted is available [here](https://huggingface.co/OpenFold/OpenFold3/tree/main/examples/multimer_precomputed_msa) 

```bash
run_openfold predict \
    --query_json /path/to/query_precomputed.json \
    --use_msa_server=False \
    --output_dir /path/to/output/ \
    --runner_yaml /path/to/inference_precomputed.yml
```

(323-inference-without-msas)=
#### 3.2.3 🚫 Inference Without MSAs
To run OpenFold3 without MSA features, you need to provide a "dummy" MSA file that only contains the query sequence. See the {ref}`MSA formatting section <1-precomputed-msa-files>` in our precomputed MSA documentation for how to prepare these files. Note that currently, if no MSAs are provided at all, the input sequence will not be propagated to the MSA embedder and prediction quality will be significantly reduced, so a completely MSA-free inference mode is currently discouraged if the goal is to obtain decent quality structures without aligned sequences. We are working on automatic dummmy MSA generation to make it the default no-MSA behavior.

(33-customized-inference-settings-using-runneryml)=
### 3.3 Customized Inference Settings Using `runner.yml`

OpenFold3 provides extensive customization options through a `runner.yml` configuration file. This file allows you to override the default settings defined in [`validator.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py) and customize the inference behavior to your needs.

We provide several example runner files in our [examples directory](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_runner_yamls) that demonstrate common use cases like:

- Running on multiple GPUs
- Using low memory settings
- Customizing output formats
- Enabling cuEquivariance kernels
- Enabling PAE (predicted aligned error) calculations
- Saving MSA and Template processing outputs
- And more

For a complete reference of all available configuration options:

- See our [full configuration example](https://github.com/aqlaboratory/openfold-3/tree/main/examples/reference_full_config/full_config.yml) with all possible settings
- Read the detailed [configuration reference documentation](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/configuration_reference.md) that explains each setting

**Important Note on Model Parameter Customization:**

The default settings of the model are defined in [`openfold3/projects/of3_all_atom/config/model_config.py`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/model_config.py#L87)

The model parameters section of the configuration may by passing an update in the `runner.yml` under the `custom` field in `model_update`. See the [`cuequivariance.yml` example](https://github.com/aqlaboratory/openfold-3/blob/main/examples/example_runner_yamls/cuequivariance.yml) for one such update. 


Note that CLI arguments take precedence over configuration file settings.

Below we'll walk through some of the most common configuration scenarios and how to implement them:


(inference-run-on-multiple-gpus)=
#### 🖥️ Run on Multiple GPUs or Nodes
The inference pipeline (as well as training) is backed by[ `Pytorch Lightning`](https://lightning.ai/docs/pytorch/stable/), which allows us to automatically distribute large batch jobs of multiple predictions across all available GPUs and nodes. To enable distributed inference, specify the hardware configuration under [`pl_trainer_args`](https://github.com/aqlaboratory/openfold-3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L141) in your `runner.yml`:

```yaml
pl_trainer_args:
  devices: 4      # Default: 1
  num_nodes: 1    # Default: 1
```

---

(custom-random-seeds-inference)=
#### 🌱 Change the random seeds for the model 

By default, only 1 random model seed is used with 5 diffusion samples, in which case, one inference run will be performed, yielding 5 output structures using the same seed.

Given `n` queries, `m` seeds and `l` diffusion samples, the model will perform `n × m` independent forward passes and produce `n × m × l` predicted structures.

A custom list of random seeds can be provided to the `runner.yml` under [`experiment_settings`](https://github.com/aqlaboratory/openfold-3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L120) in the following format

```yaml
experiment_settings:
  seeds:
    - 100
    - 101
    - ... 
```

Seeding behavior is controlled in the following priority:
- Command line argument `--num_model_seeds`
- `runner.yml` via the `experiment_settings.seeds` field.

---

#### 📦 Output in PDB Format
Change the structure output format from `cif` to `pdb` using [`output_writer_settings`](https://github.com/aqlaboratory/openfold-3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L170):
```yaml
output_writer_settings:
  structure_format: pdb    # Default: cif
```

---

(inference-low-memory-mode)=
#### 🧠 Low Memory Mode
To run inference on larger queries to run on limited memory, add the following to apply the [model presets](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/model_setting_presets.yml) to run in low memory mode.

Note: These settings cause the pairformer embedding output from the diffusion samples to be computed sequentially. Significant slowdowns may occur, especially for large number of diffusion samples.
```yaml
model_update:
  presets:
    - predict  # required for inference
    - low_mem
    - pae_enabled
```

---

#### 📊 Toggle PAE head model

Predicted Aligned Error (PAE) is a predicted confidence metric from the OpenFold3 model that is used to compute predicted TM scores. You can find more information about confidence metrics [here](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/confidence-scores-in-alphafold-multimer/).

The PAE model head is enabled by default in inference by its selection in the model presets. The current models available from OpenFold all require `pae_enabled` as a setting.

To disable PAE model, provide a list of model presets that does not include the `pae_enabled` preset, e.g.

```yaml
model_update:
  presets:
    - predict # required for inference 
```

Conversely, if you provide your own model_update and wish to use the PAE head, please ensure the `pae_enabled` preset is selected:

```yaml
model_update:
  presets:
    - predict  # required for inference
    - low_mem  # default low memory settings
    - pae_enabled  # required to run PAE head
  custom:
    - ... custom model changes
```

---

### 3.4 Customized ColabFold MSA Server Settings Using `runner.yml` 

All settings for the ColabFold server and outputs can be set under [`msa_computation_settings`](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/data/tools/colabfold_msa_server.py#L904)


(34-saving-msa-outputs)=
#### Saving MSA outputs

By default, MSA outputs are written to a temporary directory and are deleted after prediction is complete. 

These settings can be saved by changing the following fields:

```yaml
msa_computation_settings:
  msa_output_directory: <custom path>
  cleanup_msa_dir: False  # If False, msa paths will not be deleted between runs 
  save_mappings: True 
```

MSAs per chain are saved using a file / directory name that is the hash of the sequence. Mappings between the chain name, sequence, and representative ids can be saved via the `save_mappings` field. 

---

#### Use a Privately Hosted ColabFold MSA Server
Specify the URL of your private MSA server with the `server_url` field:
```yaml
msa_computation_settings:
  server_url: https://my.private.colabfold.server
```

---

#### Save MSAs in A3M Format
Choose the file format for saving MSAs retrieved from ColabFold:
```yaml
msa_computation_settings:
  msa_file_format: a3m     # Options: a3m, npz (default: npz)
```

## 4. Model Outputs

In the inference pipeline, we generate a dedicated output directory for each query, named by the corresponding query key (e.g., `query_1` or `3hfm`, if the PDB ID is provided). Each such directory will contain prediction results, MSAs, and intermediate files for MSA and template processing:

### 4.1 Prediction Outputs (`query/seed/`)

Each seed produces `l` (number of diffusion samples) structure predictions, and their associated confidence scores, stored in subdirectories named after the query, seed and the index of the diffusion sample, e.g.:
```bash
<output_directory>
 ├── query_1
	 └── seed_42
        ├── query_1_seed_42_sample_1_model.cif
        ├── query_1_seed_42_sample_1_confidences.json
        ├── query_1_seed_42_sample_1_confidences_aggregated.json
        └── timing.json 
```

- `*_model.cif` (or `.pdb`): Final predicted 3D structure (with per-atom pLDDT in B-factor if `.pdb`).
  
- `*_confidences.json`: Per-atom confidence scores:

  - `plddt`: Predicted Local Distance Difference Test

  - `pde`: Predicted Distance Error

- `*_confidences_aggregated.json`: Aggregated metric:

  - `avg_plddt` - Average pLDDT over structure

  - `gpde` - Global Predicted Distance Error (see AF3 SI Section 5.7 Eq. 16)

  The following metrics are available only when `pae_enabled` is set. 
  
  - `ptm` - Predicted TM score of a full complex (SI §5.9.1)

  - `iptm` - Interface variant of a predicted TM score of a full complex (SI §5.9.1)

  - `disorder` - Average relative solvent accessible surface area (RASA) over all protein atoms (§5.9.3, item 1)

  - `has_clash` - Whether any pair of polymer chains has steric clashes (0.0 if no clashes, 1.0 otherwise)

  - `sample_ranking_score` - Weighted sum of `ptm`, `iptm`, `disorder`, `has_clash`. Used to rank predictions (SI §5.9.3, item 1)

  - `chain_ptm` - Per-chain predicted TM score (SI §5.9.1)
  
  - `chain_pair_iptm` - Interface variant of predicted TM score of each chain pairs (SI §5.9.1, §5.9.3, item 3)
  
  - `bespoke_iptm` - Average `chain_pair_iptm` between each chain of a pair and all other chains. Used to rank interface predictions (SI § 5.9.3, item 3)

- `timing.json`: The runtime for the submitted query (s), not including the runtime for any MSA computations.


### 4.2 Processed MSAs (`main/` and `paired/`)
Only created if `--use_msa_server=True`. <br/>
Processed MSAs for each unique chain are saved as `.npz` files used to create input features for OpenFold3. 
If a chain is reused across multiple queries, its MSA is only computed once and named after the first occurrence. This reduces the number of queries to the ColabFold server.


For a sequence with two representative chains, the final output directory would have this format:

```bash
<msa_output_directory>
├── main
│   ├── <hash of sequence A>.npz
│   └── <hash of sequence B>.npz
├── mappings
│   ├── chain_id_to_rep_id.json
│   ├── query_name_to_complex_id.json
│   ├── README.md
│   ├── rep_id_to_seq.json  # hash to sequence mapping
│   └── seq_to_rep_id.json
└── paired
    └── <hash of concatenation of sequences A and B>
        ├── <hash of sequence A>.npz
        └── <hash of sequence B>.npz
```


If a query is a heteromeric protein complex (has at least two different protein chains) and `--use_msa_server` is enabled, **paired MSAs** are also generated. 
If a set of chains with a specific stoichiometry is reused across multiple queries, for example if the same heterodimer is screened against multiple small molecule ligands, its set of paired MSAs is only computed once and named after the first occurrence. This reduces the number of queries to the ColabFold server. 

```bash
<msa_output_directory>
 ├── paired
    └── <hash of concatenation of sequences A and B> 
        ├── <hash of sequence A>.npz
        └── <hash of sequence B>.npz
```

In summary, we submit a total of 1 + n queries to the ColabFold MSA server per run - one query for the set of all unqiue protein sequences in the inference query json file (unpaired/main MSAs) and n additional queries for the collection of unqiue protein chain combinations for heteromeric complexes (paired MSAs).

The MSA deduplication behavior is also present for precomputed MSAs. See the {ref}`chain deduplication utility <4-msa-reusing-utility>` section for details.

Note: The raw ColabFold MSA `.a3m` alignment files and scripts are saved to `<msa_output_directory>/raw/`. <br/> 
This directory is then deleted upon completion of MSA processing by the OpenFold3 workflow to avoid disruption to future inference submissions. <br/>

To manually keep the raw ColabFold outputs, remove this line here [here](https://github.com/aqlaboratory/openfold-3/blob/9d3ff681560cdd65fa92f80f08a4ab5becaebf87/openfold3/core/data/tools/colabfold_msa_server.py#L933). <br/>

### 4.3 Mapping outputs (`mapping/`)

If the same `msa_output_directory` is used between runs, the `rep_id_to_seq.json` and `seq_to_rep_id.json` mappings are updated with the new sequences, while the other mappings are overwritten.

```bash
<msa_output_directory>
 ├── paired
    └── <hash of concatenation of sequences A and B> 
        ├── <hash of sequence A>.npz
        └── <hash of sequence B>.npz
```

### 4.4 Query Metadata 
There are several system-generated files that record the state of submitted inference job.

- [Inference Query Set](441-inference-query-set-json) -- Input query and references to auxiliary files (e.g. MSA alignments and template files)
- [Model Config](442-model-config-json) - Model settings, e.g. architecture and memory settings
- [Experiment Config](443-experiment-config-json) -- Experiment settings for the inference run

(441-inference-query-set-json)=
#### 4.4.1 Inference Query Set (`inference_query_set.json`)
This file representing the full input query in a validated internal format defined by [this Pydantic schema](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/inference_query_format.py).

- Created automatically from the original `query.json`.

- If `--use_msa_server=True`, automatically populates:

  - `main_msa_file_paths`: Paths to single-chain `.a3m` or `.npz` files

  - `paired_msa_file_paths`: Paths to paired `.a3m` or `.npz` files (if heteromer input)

  Note: Refer to the {doc}`Precomputed MSA Documentation <precomputed_msa_how_to>` for how to specify these fields if you want to use precomputed MSAs instead of MSAs from the Colabfold server.

- If `--use_templates=True`, automatically populates:

  - `template_alignment_file_path`: Path to the preprocessed template cache entry `.npz` file used for template featurization. By default, template cache entries are automatically created in a short preprocessing step using the raw template alignment files provided under this same field and the template structures identified in the alignment. 

  - `template_entry_chain_ids`: List of template chains, identified by their entry (typically PDB) IDs and chain IDs, used for featurization. By default, up to the first 4 of these chains are used.

  Note: Refer to the {doc}`Template How-To Documentation <template_how_to>` for how to specify these fields if you want to use precomputed template alignments instead of Colabfold alignments for template inputs.

Note: If MSA and template files are persisted between runs, the same `inference_query_set.json` file can be used to resubmit the query without needing to rerun the template and MSA pipelines. To do so:

1. Turn off the [MSA cleanup option](34-saving-msa-outputs).
2. pass in the generated `inference_query_set.json` as the `query.json` and use `--use_msa_server=False` and `--use_templates=True`.

Model seeds should still be set either from the command line or using the `seeds` field under `experiment_settings` in the `runner.yml`.

(442-model-config-json)=
#### 4.4.2 Model Config (`model_config.json`)

This file represents the model settings used to perform inference. The config follows the model configuration file defined [here](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/projects/of3_all_atom/config/model_config.py)
This file represents the model settings used to perform inference. The config follows the model configuration file defined [here](../../openfold3/projects/of3_all_atom/config/model_config.py#L71).

(443-experiment-config-json)=
#### 4.4.3 Experiment Config (`experiment_config.json`)

This file records the entire state of the experiment, as defined by the [InferenceExperimentConfig pydantic model](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/entry_points/validator.py#L347).


**🔗 Example:**

See the full multimer output for [Deoxy human hemoglobin](https://huggingface.co/OpenFold/OpenFold3/tree/main/examples/output_multimer_with_colabfold_msas), which were generated with `run_multimer.sh`. For this example, the colabfold_msas and colabfold_template directories are stored in the same outbut directory. 


When processing multimer inputs (e.g., hemoglobin α + β chains), OpenFold3 automatically:

- Requests paired MSAs from the ColabFold server
- Stores raw alignments in `raw/paired/` temporarily
- Converts them into per-chain `.npz` alignments in [`paired/`](https://huggingface.co/OpenFold/OpenFold3/tree/main/examples/output_multimer_with_colabfold_msas/colabfold_msas/paired)




