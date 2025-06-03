# OpenFold Inference

In this guide, we will cover how to use OpenFold3 to make structure predictions.

### Pre-requisites: 

- OpenFold3 Conda Environment. See [OpenFold3 Installation](installation.md) for instructions on how to build this environment.
- OpenFold3 Model Parameters.


## Running OpenFold3 Inference

A directory containing containing multiple inference examples is provided [here](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3). These include:
- [Single-chain protein (monomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/monomer): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/homomer): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (heteromer/multimer)](https://github.com/aqlaboratory/openfold3/tree/main/examples_of3/multimer): Deoxy humman hemoglobin (PDB: 1A3N)
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

The expected output contents are as follows:
~~ TO ADD ~~


### Changing Default Inference Workflow