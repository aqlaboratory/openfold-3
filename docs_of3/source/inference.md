# OpenFold Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 carefully implements the features described in AlphaFold 3 *Nature* paper.

This guide covers how to use OpenFold3 to make structure predictions.


## 1. Inference features

OpenFold3 replicates the full set of input features described in the *AlphaFold 3* publication. All of these features are **fully implemented and supported in training mode**. We are actively working on integrating these functionalities into the inference pipeline. 
 
Below is the current status of inference feature support by molecule type:


### 1.1. Protein

Supported:

- Prediction with MSA, using ColabFold MSA pipeline
- Prediction without MSA

Coming soon:

- OpenFold3's own MSA generation pipeline
- Support for OpenFold3-style precomputed MSAs
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


### 3.2. Inference without Pre-computed Alignments

The following command performs model inference using the ColabFold server for MSA generation. <br/>
Integration of pre-computed MSAs or OpenFold3s internal MSA generation into the inference pipeline will be supported in the full codebase release.

```
python run_openfold predict \
    --query_json /path/to/inference/query.json \
    --inference_ckpt_path /path/inference.ckpt
    --use_msa_server \
    --output_dir /path/output \
```

**Required arguments:**

- `--query_json` *(Path)*: Path to the JSON file specifying input sequences to predict and metadata.

- `--inference_ckpt_path` *(Path)*: Path to the model checkpoint.

- `--use_msa_server` *(bool, default = True)*: Use ColabFold MSA server to create alignments. This is required in the current preliminary inference release.


**Optional Inference Arguments:**

- `--runner_yaml` *(Path)*: YAML config specifying model and data parameters. For full control over settings, edit this file directly. Example: [runner.yml](https://github.com/aqlaboratory/openfold3/tree/main/examples/runner.yml).

- `--output_dir` *(Path)*: Directory where outputs will be written. Defaults to `test_train_output/`

- `--num_diffusion_samples` *(int, default = None)*: Number of diffusion samples per query. If unspecified, defaults to 5 diffusion samples.

- `--num_model_seeds` *(int, default = None)*: Number of model seeds to use per query. If unspecified, defaults to one seed (42).

These flags allow you to customize the inference workflow. As for all OpenFold3 parameters, `output_dir`, `num_diffusion_samples` and `num_model_seeds` can both also be updated directed via `runner.yml`.


## 4. Model Outputs

OpenFold3's output format currently follows the structure used by the ColabFold server. During processing, chain IDs are internally mapped to standardized identifiers, then re-mapped back to the original query IDs in the final outputs.

For each unique chain, a single MSA is generated and saved as an `.npz` file. If a chain appears multiple times across different queries, it is deduplicated — the MSA will be stored under the name of its first occurrence.

Each query results in an output directory with one subfolder per seed. A `main` directory stores MSAs and processed input features; a `raw` directory contains raw alignment and template search outputs.

#### Example Output Structure (Single-chain, single seed)

See [Monomer Output Example](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/monomer) for full details.

```
query.json
<output_directory_path>
 ├── <protein.
	 └── seed_42
 ├── main
	 └── <protein_msa>.npz
└──  raw
	 └── main
	     ├── bfd.mgnify30.metaeuk30.smag30
	     ├── msa.sh
	     ├── out.tar.gz
	     ├── pdb70.m8
         ├── uniref.a3m
         └── templates_101
             ├── 1dlr.cif
             ├── 1dr5.cif
             ├── pdb70_cs219.ffdata
             ├── pdb70_cs219.ffindex --> pdb70_a3m.ffindex
             ├── pdb70_a3m.ffdata
             └── pdb70_a3m.ffindex
```

#### Paired MSA outputs
When the input contains multiple distinct chains, additional paired alignment files are generated under a `paired/` directory. </br>
See the [Multimer Output Example](examples_of3/multimer) for a complete case.

```
paired
 ├── pair.a3m
 ├── pair.sh
 └── out.tar.gz
```
