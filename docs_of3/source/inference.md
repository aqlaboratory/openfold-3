# OpenFold Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 carefully implements the features described in AlphaFold 3 *Nature* paper.

This guide covers how to use OpenFold3 to make structure predictions.


## 1. Inference features

OpenFold3 replicates the full set of input features described in the *AlphaFold 3* publication. All of these features are **fully implemented and supported in training mode**. We are actively working on integrating these functionalities into the inference pipeline. 
 
Below is the current status of inference feature support by molecule type:


### 1.1. Protein

Supported:

- Prediction with MSA, using ColabFold MSA pipeline or pre-computed MSAs
- Prediction without MSA

Coming soon:

- OpenFold3's own MSA generation pipeline
- Template-based prediction
- Non-standard or covalently modified residues
- Pocket conditioning *(requires fine-tuning)*

### 1.2. DNA

Supported:

- Prediction without MSA (per AF3 default)

Coming soon:

- Non-standard or covalently modified residues


### 1.3. RNA

Supported:

- Prediction without MSA

Coming soon:

- OpenFold3's own MSA generation pipeline
- Support for OpenFold3-style precomputed MSAs
- Non-standard or covalently modified residues


### 1.4. Ligand

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


### 3.1. Input Data

Queries can include any combination of single- or multi-chain proteins, with or without ligands, and may contain multiple such complexes. <br/>
See [OpenFold3 input format](input_format.md) for instructions on how to specify your input data.


### 3.2. Default Inference settings

OpenFold3 currently supports two inference modes:

- **Inference with on-the-fly MSA generation** using the ColabFold MSA server (default)
- **Inference with precomputed MSAs** using custom `.npz` files and/or local `.a3m` alignments


#### 3.2.1. 🚀 Inference with ColabFold MSA Server (default)

Use this mode to generate MSAs automatically via the ColabFold server:
```
python run_openfold.py predict \
    --query_json /path/to/inference/query.json \
    --inference_ckpt_path /path/to/inference.ckpt \
    --use_msa_server \
    --output_dir /path/to/output/
```

🧪 *Notes*: 
- Internal OpenFold3 MSA generation will be supported in future releases.
- Only the protein sequences that are provided are sent to the colabfold MSA server when the `--use_msa_server` option is selected


#### 3.2.2. 💾 Inference with Precomputed MSAs

Use this mode when you have manually prepared .npz and .a3m MSA files:
```
python run_openfold.py predict \
    --query_json /path/to/query_precomputed_full_path.json \
    --use_msa_server=False \
    --inference_ckpt_path /path/to/of3_v14_79-32000_converted.ckpt.pt \
    --output_dir /path/to/precomputed_prediction_output/ \
    --runner_yaml /path/to/inference_precomputed.yml
```

#### 3.2.3. ⚙️ Inference Arguments

**Required**

- `--query_json` *(Path)*`
    - Path to the input query JSON file.

- `--inference_ckpt_path` *(Path)*
    - Path to the model checkpoint file (e.g., .pt file).


**Optional**
- `--runner_yaml` *(Path)*
    - YAML config for full control over model and data settings. Example: [runner.yml](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml)

- `--output_dir` *(Path)*
    - Output directory for all results. Defaults to test_train_output/.

- `--use_msa_server` *(bool, default = True)*
    - Whether to generate MSAs via the ColabFold server.


- `--num_diffusion_samples` *(int, default = 5)*
    - Number of diffusion samples per query (default is `5 samples`).

- `--num_model_seeds` *(int, default = 42)*
    - Number of random seeds to use per query (default is seed `42`).

🔁 These settings can also be defined in `runner_yaml`, overriding command-line defaults.


### 3.3. Customized inference settings 

You can customize inference behavior by providing a [`runner.yml`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/examples/runner_inference.yml) file. This overrides the default settings defined in [`validator.py`](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/entry_points/validator.py).

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

During inference, internal chain identifiers are used temporarily and then mapped back to the original query-defined `chain_ids` in the final output files.

---

### 4.1. 🔄 Output Directory Layout

Each query result is stored in:

```
<output_directory_path>
 ├── query_1
	 └── seed_42
        ├── query_1_seed_42_sample_1_model.cif
        ├── query_1_seed_42_sample_1_confidences.json
        └── query_1_seed_42_sample_1_confidences_aggregated.json
 ├── main
	 └── query_1.npz
 ├──  raw
	 └── main
	     ├── bfd.mgnify30.metaeuk30.smag30
	     ├── msa.sh
	     ├── out.tar.gz
	     ├── pdb70.m8
         └── uniref.a3m
 └── inference_query_set.json
```
### 4.2. 📁 Output Components

#### **Prediction Subdirectory (`query_1/seed_42/`)**
Each seed produces one or more sampled structure predictions and their associated confidence scores:

- `*_model.cif` (or `.pdb`): Predicted 3D structure
  - If PDB format is selected, per-atom pLDDT scores are embedded.
  
- `*_confidences.json`: Per-atom confidence metrics
  - `plddt`: Predicted Local Distance Difference Test
  - `pde`: Predicted Distance Error

- `*_confidences_aggregated.json`: Summary metrics
  - `avg_plddt`: Average pLDDT over structure
  - `gPDE`: Global Predicted Distance Error (see Eq. 16 in AF3 SI Section 5.7)

#### **MSA and Input Feature Directory (`main/`)**
- Contains a single `.npz` file per unique chain (deduplicated across queries).
- File is named after the first occurrence of that chain across queries.

#### **Raw ColabFold MSA Outputs (`raw/main/`)**
- Includes intermediate alignment files and scripts generated by the ColabFold MSA server.
- Only generated if `use_msa_server=True`.

#### **Query Tracking (`inference_query_set.json`)**
- A system-generated file representing the full input query in a validated internal format.
- Defined by [this Pydantic model](https://github.com/aqlaboratory/openfold3/blob/inference-dev/openfold3/projects/of3_all_atom/config/inference_query_format.py).
- If `use_msa_server` is enabled, this file includes:
  - `main_msa_file_paths`
  - `paired_msa_file_paths`
  pointing to the relevant ColabFold-generated `.a3m` files.


### 4.3. 🧬 Multimer Prediction Example

When multiple chains are defined in a query, paired MSAs are also generated. These appear under a `paired/` directory:

```
paired
 ├── pair.a3m
 ├── pair.sh
 └── out.tar.gz
```

For a full multimer example, see: [deoxy human hemoglobin](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/multimer).





