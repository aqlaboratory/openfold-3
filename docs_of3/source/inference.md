# OpenFold Inference

Welcome to the Documentation for running inference with OpenFold3, our fully open source, trainable, PyTorch-based reproduction of DeepMind’s AlphaFold 3. OpenFold3 carefully implements the features described in AlphaFold 3 *Nature* paper.

This guide covers how to use OpenFold3 to make structure predictions.


## Inference features

 OpenFold3 replicates the full set of input features described in AlphaFold 3 publication. All of these features are *fully implemented and supported in training mode*.

 However, in this preliminary release of the inference pipeline, a few of these features are not yet exposed for inference use. These include:

- OpenFold3's built-in MSA generation pipeline
- Custom user-provided MSAs
- Nucleic acids
- Covalently bound ligands

These features will be fully integrated into the inference workflow in the final codebase release.


## Pre-requisites:

- OpenFold3 Conda Environment. See [OpenFold3 Installation](installation.md) for instructions on how to build this environment.
- OpenFold3 Model Parameters.


## Running OpenFold3 Inference

A directory containing containing multiple inference examples is provided [here](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3). These include:
- [Single-chain protein (monomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/monomer): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/homomer): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (heteromer/multimer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/multimer): Deoxy human hemoglobin (PDB: 1A3N)
- [Protein-ligand complex](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/protein_ligand_complex): Mcl-1 with small molecule inhibitor (PDB: 5FDR)
- [Multiple distinct proteins]()


### Input Data

Queries can include any combination of single- or multi-chain proteins, with or without ligands, and may contain multiple such complexes. <br/>
See [OpenFold3 input format](input_format.md) for instructions on how to specify your input data.


### Inference without Pre-computed Alignments

The following command performs model inference using the MMseqs server for MSA generation. <br/>
**Note**: The current lightweight version only supports MMseqs-based MSAs, but future versions will aditionally support using pre-computed MSAs or MSAs generated via OpenFold3s internal pipeline.

```
python run_openfold predict \
    --use_msa_server \
    --query_json /path/to/inference/query.json \
    --output_dir /path/output \
	--inference_ckpt_path /path/inference.ckpt
```

**Required arguments:**
- `--use_msa_server`: Use MMseqs server to create alignments.
- `--query_json`: JSON file of input data, containing the sequence(s) to predict.
- `--output_dir`: Specify the output directory.
- `--inference_ckpt_path`: Path to the model weights.


### Model Outputs

#### For single & multiple identical protein chains:
The expected output contents are as follows:
- `query.json`
- `output/`:
    - `<protein_name>/`
        - `seed/`
    - `main/`
        - `protein_msa.npz`
    - `raw/`
        - `main/`
            - `bfd.mgnify30.metaeuk30.smag30`
            - `msa.sh`
            - `out.tar.gz`
            - `pdb70.m8`
            - `uniref.a3m`
    - `inference_query_set.json`

where:
- `<protein_name>/seed/`
- `main/`
- `raw/`
- `inference_query_set.json`

#### For distinct protein chains (multimers)
For multimers, the output directory will additionally contain a `paired` MSA directory, containing the MSA for the distinct single protein chains.
As an example, the output directory for the [deoxy human hemoglobin](examples_of3/multimer), which comprises two distinct chains `A` and `B`, e.g , the output will be as follows: 

- `query.json`
- `output/`:
    - `hemoglobin>/`
    - `main/`
    - `raw/`
    - `paired/`
        - `chainA.chainA.chainB.chainB/`
            - `chainA.npz`
            - `chainB.npz`
    - `inference_query_set.json`



### Changing Default Inference Workflow