# OpenFold Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 carefully implements the features described in [AlphaFold 3 *Nature* paper](https://www.nature.com/articles/s41586-024-07487-w).

This guide covers how to use OpenFold3 to make structure predictions.


## 1. Inference features

OpenFold3 replicates the full set of input features described in the *AlphaFold 3* publication. All of these features are **fully implemented and supported in training mode**. We are actively working on integrating these functionalities into the inference pipeline. 
 
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

### 1.2 DNA

Supported:

- Prediction without MSA (per AF3 default)
- Non-canonical residues

Coming soon:

- Covalently modified residues and other cross-chain covalent bonds


### 1.3 RNA

Supported:

- Prediction without MSA
- OpenFold3's own MSA generation pipeline
- Support for OpenFold3-style precomputed MSAs
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

- OpenFold3 Conda Environment. See [OpenFold3 Installation](installation.md) for instructions on how to build this environment.
- OpenFold3 Model Parameters: please find the checkpoints [in this Google Drive](https://drive.google.com/drive/folders/1PD1B-FuLF9V9wxATGh7qaF0G-WaT4j3g?usp=drive_link).


## 3. Running OpenFold3 Inference

A directory containing containing multiple inference examples is provided [in this Google Drive](https://drive.google.com/drive/folders/1b4OXHxXUdSd-XYrqtblIF-64rt9Mda4Q?usp=drive_link). These include:
- [Single-chain protein (monomer)](https://drive.google.com/drive/folders/15S0Z_EIj5JJ4eWUaMi3uhCIhl4TzIWgN?usp=drive_link): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://drive.google.com/drive/folders/1mxuhRij04bZu6D5UtxlEjl8n6LU5hwrA?usp=drive_link): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (multimer)](https://drive.google.com/drive/folders/1d0S6ueEyrUEMVeXiebDhEeTp6s-Hm_Pu?usp=drive_link): Deoxy human hemoglobin (PDB: 1A3N)
- [Protein-ligand complex](https://drive.google.com/drive/folders/1MUYcp-EN1JizM1-xgz94qx5pcVS1A_Ap?usp=drive_link): Mcl-1 with small molecule inhibitor (PDB: 5FDR)


### 3.1 Input Data

Queries can include any combination of single- or multi-chain proteins, with or without ligands, and may contain multiple such complexes. <br/>
Input is provided via a `query.json` file — a structured JSON document that defines each query, its constituent chains, chain types (e.g., protein, DNA, ligand) and sequences or molecular graphs. Optionally, the query can include paths to precomputed protein or RNA MSAs. <br/>
See [OpenFold3 input format](input_format.md) for instructions on how to specify your input data.


### 3.2 Inference Modes
OpenFold3 currently supports three inference modes:

- 🚀 With ColabFold MSA Server (default)
- 📂 With Precomputed MSAs
- 🚫 Without MSAs (MSA-free)

Each mode shares the same command structure but differs in how MSAs are provided or generated.

#### 3.2.1 🚀 Inference with ColabFold MSA Server (Default)

This mode automatically generates MSAs using the ColabFold server. Only protein sequences are sent to the server. We recommend this mode if you only have a couple of structures to predict.

```
python run_openfold.py predict \
    --query_json /path/to/query.json \
    --inference_ckpt_path /path/to/inference.ckpt \
    --use_msa_server \
    --output_dir /path/to/output/
```

**Required arguments**

- `--query_json` *(Path)*
    - Path to the input query JSON file.

- `--inference_ckpt_path` *(Path)*
    - Path to the model checkpoint file (`.pt` file).


**Optional arguments**

- `--use_msa_server` *(bool, optional, default = True)*
    - Whether to use the ColabFold server for MSA generation.

- `--output_dir` *(Path, optional, default = `test_train_output/`)*
    - Directory where outputs will be written.

- `--num_diffusion_samples` *(int, optional, default = 5)*
    - Number of diffusion samples per query.

- `--num_model_seeds` *(int, optional, default = 42)*
    - Number of random seeds to use per query.

- `--runner_yaml` *(Path, optional, default = null)*
    - YAML config for full control over model and data parameters.
    - Example: [runner.yml](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml)

📝  *Notes*: 
- Only protein sequences are submitted to the ColabFold server so this mode only uses MSAs for protein chains.
- All arguments can also be set via `runner_yaml`, but command-line flags take precedence and will override values specified in the YAML file (see [Customized Inference Settings](#33-customized-inference-settings-using-runneryml) for details).


#### 3.2.2 📂 Inference with Precomputed MSAs
This mode allows inference using MSA files prepared manually or by external tools. We recommend this mode for high-throughput screeing applications where you want to run hundreds or thousands of predictions. See the [precomputed MSA documentation](precomputed_msa_how_to.md) for a step-by-step tutorial, the [MSA generation guide](precomputed_msa_generation_how_to.md) for using our MSA generation pipeline and the [precomputed MSA explanatory document](precomputed_msa_explanation.md) for a more in-depth explanation on how precomputed MSA handling works.

```
python run_openfold.py predict \
    --query_json /path/to/query_precomputed.json \
    --inference_ckpt_path /path/to/of3_checkpoint.pt \
    --use_msa_server=False \
    --output_dir /path/to/output/ \
    --runner_yaml /path/to/inference_precomputed.yml
```

#### 3.2.3 🚫 Inference Without MSAs
This mode skips MSA generation entirely. OpenFold3 will perform inference using only the input sequences. Prediction quality will be reduced compared to MSA-based modes. This inference mode is currently discouraged if the goal is to obtain the highest-accuracy structures.

```
python run_openfold.py predict \
    --query_json /path/to/query.json \
    --inference_ckpt_path /path/to/inference.ckpt \
    --use_msa_server=False \
    --output_dir /path/to/output/
```

### 3.3 Customized Inference Settings Using `runner.yml`

You can further customize inference behavior by providing a [`runner.yml`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml) file. This overrides the default settings defined in [`validator.py`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/entry_points/validator.py).

Below are some common use cases and how to configure them:

---

#### 🖥️ Run on Multiple GPUs or Nodes
Specify the hardware configuration under [`pl_trainer_args`](https://github.com/aqlaboratory/openfold3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L141) in `runner.yml`:

Note: Using multiple GPUs in combination with the `--use_msa_server` option currently launches the same ColabFold MSA server query and template preprocessing code per GPU. It may be more efficient to pre-compute the MSAs and preprocess templates in advance and then running distributed predictions with the [pre-computed MSA option](precomputed_msa_how_to.md). We will introduce a fix to this in an upcoming release.
```
pl_trainer_args:
  devices: 4      # Default: 1
  num_nodes: 1    # Default: 1
```

---

#### 📦 Output in PDB Format
Change the structure output format from `cif` to `pdb` using [`output_writer_settings`](https://github.com/aqlaboratory/openfold3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L170):
```
output_writer_format:
  structure_format: pdb    # Default: cif
```

---

#### 🧠 Low Memory Mode
To run inference on larger queries to run on limited memory, add the following to apply the [model presets](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/projects/of3_all_atom/config/model_setting_presets.yml) to run in low memory mode.

Note: These settings cause the pairformer embedding output from the diffusion samples to be computed sequentially. Significant slowdowns may occur, especially for large number of diffusion samples.
```
model_update:
  presets:
    - predict  # required for inference
    - low_mem
```

---

### 3.4 Customized ColabFold MSA Server Settings Using `runner.yml` 

All settings for the ColabFold server and outputs can be set at `msa_computation_settings`](https://github.com/aqlaboratory/openfold3/blob/9d3ff681560cdd65fa92f80f08a4ab5becaebf87/openfold3/core/data/tools/colabfold_msa_server.py#L833)


#### Saving MSA outputs

By default, MSA outputs are written to a temporary directory and are deleted after prediction is complete. 

These settings can be saved by changing the following fields:

```
msa_computation_settings:
  msa_output_directory: <custom path>
  cleanup_msa_dir: False  # If False, msa paths will not be deleted between runs 
  save_mappings: True 
```

MSAs per chain are saved using a file / directory name that is the hash of the sequence. Mappings between the chain name, sequence, and representative ids can be saved via the `save_mappings` field. 

---

#### Use a Privately Hosted ColabFold MSA Server
Specify the URL of your private MSA server with the `server_url` field:
```
msa_computation_settings:
  server_url: https://my.private.colabfold.server
```

---

#### Save MSAs in A3M Format
Choose the file format for saving MSAs retrieved from ColabFold:
```
msa_computation_settings:
  msa_file_format: a3m     # Options: a3m, npz (default: npz)
```

## 4. Model Outputs

OpenFold3 produces a structured set of outputs modeled after the ColabFold server. Each query in the input json file (e.g., `query_1`) generates a dedicated output directory containing prediction results, MSAs, and intermediate files, for instance for template processing.

During processing, chain IDs are mapped to internal standardized names, then re-mapped back to the original query IDs (`chain_ids`) in the final output files.

Each query produces a structured output directory with the following components:

### 4.1 Prediction Outputs (`query/seed/`)

Each seed produces one or more sampled structure predictions and their associated confidence scores, stored in subdirectories named after the query and seed, e.g.:
```
<output_directory>
 ├── query_1
	 └── seed_42
        ├── query_1_seed_42_sample_1_model.cif
        ├── query_1_seed_42_sample_1_confidences.json
        └── query_1_seed_42_sample_1_confidences_aggregated.json
```

- `*_model.cif` (or `.pdb`): Final predicted 3D structure (with per-atom pLDDT in B-factor if `.pdb`).
  
- `*_confidences.json`: Per-atom confidence scores:

  - `plddt`: Predicted Local Distance Difference Test

  - `pde`: Predicted Distance Error

- `*_confidences_aggregated.json`: Aggregated metric:

  - `avg_plddt` - Average pLDDT over structure

  - `gpde` - Global Predicted Distance Error (see AF3 SI Section 5.7 Eq. 16)


### 4.2 Processed MSAs (`main/` and `paired/`)
Only created if `--use_msa_server=True`. <br/>
Processed MSAs for each unique chain are saved as `.npz` files used to create input features for OpenFold3. 
If a chain is reused across multiple queries, its MSA is only computed once and named after the first occurrence. This reduces the number of queries to the ColabFold server.


For a sequence with two representative chains, the final output directory would have this format:

```
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




```
<msa_output_directory>
 ├── main
    ├── <hash of query 1, sequence A>.npz
    └── <hash of query 1, sequence B>.npz
```

If a query is a heteromeric protein complex (has at least two different protein chains) and `--use_msa_server` is enabled, **paired MSAs** are also generated. 
If a set of chains with a specific stoichiometry is reused across multiple queries, for example if the same heterodimer is screened against multiple small molecule ligands, its set of paired MSAs is only computed once and named after the first occurrence. This reduces the number of queries to the ColabFold server. 

```
<msa_output_directory>
 ├── paired
    └── <hash of concatenation of sequences A and B> 
        ├── <hash of sequence A>.npz
        └── <hash of sequence B>.npz
```

In summary, we submit a total of 1 + n queries to the ColabFold MSA server per run - one query for the set of all unqiue protein sequences in the inference query json file (unpaired/main MSAs) and n additional queries for the sets of of proteins chains heteromeric complexes (paired MSAs).

The MSA deduplication behavior is also present for precomputed MSAs. See the [chain deduplication utility](precomputed_msa_explanation.md#4-msa-reusing-utility) section for details.

### 4.3 Mapping outputs (`mapping/`)

If the same `msa_output_directory` is used between runs, the `rep_id_to_seq.json` and `seq_to_rep_id.json` mappings are updated with the new sequences, while the other mappings are overwritten.

```
<msa_output_directory>
 ├── paired
    └── <hash of concatenation of sequences A and B> 
        ├── <hash of sequence A>.npz
        └── <hash of sequence B>.npz
```


#### Note: Raw ColabFold MSA Outputs
The raw ColabFold MSA `.a3m` alignment files and scripts are saved to `<msa_output_directory>/raw/`. <br/> 
This directory is then deleted upon completion of MSA processing by the OpenFold3 workflow to avoid disruption to future inference submissions. <br/>

To manually keep the raw ColabFold outputs, remove this line here [here](https://github.com/aqlaboratory/openfold3/blob/9d3ff681560cdd65fa92f80f08a4ab5becaebf87/openfold3/core/data/tools/colabfold_msa_server.py#L933). <br/>


### 4.4 Query Metadata (`inference_query_set.json`)
This is a system-generated file representing the full input query in a validated internal format defined by [this Pydantic schema](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/projects/of3_all_atom/config/inference_query_format.py).
- Created automatically from the original `query.json`.

- If `--use_msa_server=True`, includes:

  - `main_msa_file_paths`: Paths to single-chain `.a3m` files

  - `paired_msa_file_paths`: Paths to paired `.a3m` files (if heteromer input)

- If `--use_templates=True`, includes:

  - `template_alignment_file_path`: Path to the preprocessed template cache entry `.npz` file used for template featurization. By default, template cache entries are automatically created in a short preprocessing step using the raw template alignment files provided under this same field and the template structures identified in the alignment. For more details, see the [template explanatory document](template_explanation.md).

  - `template_entry_chain_ids`: List of template chains, identified by their entry (typically PDB) IDs and chain IDs, used for featurization. By default, up to the first 4 of these chains are used.

**🔗 Example:**

See the full multimer output for [Deoxy human hemoglobin](https://drive.google.com/drive/folders/1d0S6ueEyrUEMVeXiebDhEeTp6s-Hm_Pu?usp=drive_link).


When processing multimer inputs (e.g., hemoglobin α + β chains), OpenFold3 automatically:

- Requests paired MSAs from the ColabFold server
- Stores raw alignments in [`raw/paired/](https://drive.google.com/drive/folders/19CN9S3T060KahXj0wbJXMlfth8MFlyPf?usp=drive_link)
- Converts them into per-chain `.npz` alignments in [`paired/`](https://drive.google.com/drive/folders/1VAlJ6XCtt3Y434_t_vomTeMzt7S2pQdy?usp=drive_link)




