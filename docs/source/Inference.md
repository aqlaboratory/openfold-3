# OpenFold Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 carefully implements the features described in AlphaFold 3 *Nature* paper.

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

Coming soon:

- OpenFold3's own MSA generation pipeline
- Template-based prediction
- Non-standard or covalently modified residues
- Pocket conditioning *(requires fine-tuning)*

### 1.2 DNA

Supported:

- Prediction without MSA (per AF3 default)

Coming soon:

- Non-standard or covalently modified residues


### 1.3 RNA

Supported:

- Prediction without MSA

Coming soon:

- OpenFold3's own MSA generation pipeline
- Support for OpenFold3-style precomputed MSAs
- Non-standard or covalently modified residues


### 1.4 Ligand

Supported:

- Non-covalently bound ligands

Coming soon:

- Covalently bound ligands
- Polymeric ligands


## 2. Pre-requisites:

- OpenFold3 Conda Environment. See [OpenFold3 Installation](installation.md) for instructions on how to build this environment.
- OpenFold3 Model Parameters.


## 3. Running OpenFold3 Inference

A directory containing containing multiple inference examples is provided [here](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3). These include:
- [Single-chain protein (monomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/monomer): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/homomer): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (multimer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/multimer): Deoxy human hemoglobin (PDB: 1A3N)
- [Protein-ligand complex](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/protein_ligand_complex): Mcl-1 with small molecule inhibitor (PDB: 5FDR)


### 3.1 Input Data

Queries can include any combination of single- or multi-chain proteins, with or without ligands, and may contain multiple such complexes. <br/>
Input is provided via a `query.json` file — a structured JSON document that defines each query, its constituent chains, chain types (e.g., protein, DNA, ligand) and sequences. Optionally, the query can include paths to precomputed MSAs for each chain or chain pair. <br/>
See [OpenFold3 input format](input_format.md) for instructions on how to specify your input data.


### 3.2 Inference Modes
OpenFold3 currently supports three inference modes:

- 🚀 With ColabFold MSA Server (default)
- 📂 With Precomputed MSAs
- 🚫 Without MSAs (MSA-free)

Each mode shares the same command structure but differs in how MSAs are provided or generated.

#### 3.2.1 🚀 Inference with ColabFold MSA Server (Default)

This mode automatically generates MSAs using the ColabFold server. Only protein sequences are sent to the server.

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

- `--use_msa_server` *(bool, default = True)*
    - Whether to use the ColabFold server for MSA generation.

- `--output_dir` *(Path, default = `test_train_output/`)*
    - Directory where outputs will be written.

- `--num_diffusion_samples` *(int, default = 5)*
    - Number of diffusion samples per query.

- `--num_model_seeds` *(int, default = 42)*
    - Number of random seeds to use per query.

- `--runner_yaml` *(Path)*
    - YAML config for full control over model and data parameters.
    - Example: [runner.yml](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml)

📝  *Notes*: 
- Only protein sequences are submitted to the ColabFold server. 
- Internal OpenFold3 MSA generation will be supported in future releases.
- All arguments can also be set via `runner_yaml`, which overrides command-line flags (more on this [below](#33-customized-inference-settings)).


#### 3.2.2 📂 Inference with Precomputed MSAs
This mode allows inference using `.npz` or `.a3m` MSA files prepared manually or by external tools.

```
python run_openfold.py predict \
    --query_json /path/to/query_precomputed.json \
    --inference_ckpt_path /path/to/of3_checkpoint.pt \
    --use_msa_server=False \
    --output_dir /path/to/output/ \
    --runner_yaml /path/to/inference_precomputed.yml
```
📝  *Note:*
- Documentation on generating OpenFold3-compatible precomputed MSAs will be published soon.


#### 3.2.3 🚫 Inference Without MSAs
This mode skips MSA generation entirely. OpenFold3 will perform inference using only the input sequences. This is supported for proteins, DNA, and RNA inputs, though accuracy may be reduced compared to MSA-based modes.

```
python run_openfold.py predict \
    --query_json /path/to/query.json \
    --inference_ckpt_path /path/to/inference.ckpt \
    --use_msa_server=False \
    --output_dir /path/to/output/
```


### 3.3 Customized Inference Settings

You can further customize inference behavior by providing a [`runner.yml`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml) file. This overrides the default settings defined in [`validator.py`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/entry_points/validator.py).

Below are common use cases and how to configure them:

---

#### 🖥️ Run on Multiple GPUs or Nodes
Specify the hardware configuration under [`pl_trainer_args`](https://github.com/aqlaboratory/openfold3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L141) in `runner.yml`:
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

#### 🌐 Use a Privately Hosted ColabFold MSA Server
Specify the URL of your private MSA server under [`msa_server_settings`](https://github.com/aqlaboratory/openfold3/blob/aadafc70bcb9e609954161660314fcf133d5f7c4/openfold3/entry_points/validator.py#L171):
```
msa_server_settings:
  server_url: https://my.private.colabfold.server
```

---

#### 💾 Save MSAs in A3M Format
Choose the file format for saving MSAs retrieved from ColabFold:
```
msa_server_settings:
  msa_file_format: a3m     # Options: a3m, npz (default: npz)
```

## 4. Model Outputs

OpenFold3 produces a structured set of outputs modeled after the ColabFold server. Each query (e.g., `query_1`) generates a dedicated output directory containing prediction results, MSAs, and intermediate files.

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


### 4.2 Processed MSA (`main/`)

Processed MSAs for each unique chain are saved as `.npz` files. If a chain is reused across multiple queries, its MSA is only computed once and named after the first occurrence:

```
 ├── main
    └── <chain_id>.npz
```

### 4.3 Raw ColabFold MSA Outputs (`raw/`)
Only created if `--use_msa_server=True`. </br>
Include intermediate alignment files and scripts generated by ColabFold MSA server:

```
raw/
└── main/
    ├── bfd.mgnify30.metaeuk30.smag30
    ├── msa.sh
    ├── out.tar.gz
    ├── pdb70.m8
    └── uniref.a3m
```


### 4.4 Query Metadata (`inference_query_set.json`)
This is a system-generated file representing the full input query in a validated internal format defined by [this Pydantic schema](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/projects/of3_all_atom/config/inference_query_format.py).
- Created automatically from the original `query.json`.

- If `--use_msa_server=True`, includes:

  - `main_msa_file_paths`: Paths to single-chain `.a3m` files

  - `paired_msa_file_paths`: Paths to paired `.a3m` files (if multimer input)


### 4.5 Multimer Output
When multiple chains are defined in a query and `--use_msa_server=True` is enabled, **paired MSAs** are generated and stored in both processed and raw forms:

```
paired/
└── Chain-A.Chain-A.Chain-B.Chain-B
  ├── chain-A.npz
  └── chain-B.npz

raw/
└── paired/
    └── Chain-A.Chain-A.Chain-B.Chain-B
      ├── pair.a3m
      ├── pair.sh
      └── out.tar.gz
```

- `paired/`:
  - Contains processed `.npz` feature files derived from ColabFold's paired MSAs.
  - Each subdirectory is named according to the chain combination in the format: `Chain-A.Chain-A.Chain-B.Chain-B`, based on the input chain IDs provided in the query.

- `raw/paired/`:
  - Contains raw alignment output from ColabFold for each chain pair, including the original `.a3m` and associated scripts.

**🔗 Example:**

See the full multimer output for [Deoxy human hemoglobin](https://github.com/aqlaboratory/openfold3/tree/inference-dev/examples/examples/multimer/example_output/).


When processing multimer inputs (e.g., hemoglobin α + β chains), OpenFold3 automatically:

- Requests paired MSAs from the ColabFold server
- Stores raw alignments in [`raw/paired/](https://github.com/aqlaboratory/openfold3/tree/inference-dev/examples/examples/multimer/example_output/raw/paired/)
- Converts them into per-chain `.npz` features in [`paired/`](https://github.com/aqlaboratory/openfold3/tree/inference-dev/examples/examples/multimer/example_output/paired/)




