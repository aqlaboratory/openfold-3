# OpenFold3 Input Format

## 1. High-level Structure
The OpenFold3 inference pipeline takes a single JSON file as input, specifying the data and options required for structure prediction. This file can define multiple prediction targets (`queries`), which can be proteins, including individual protein chains and complexes, nucleic acids, and ligands. An example of the top-level structure of this input file is shown below:

```
{
  "queries": {
    "query_1": { ... },
    "query_2": { ... }
  },
  "ccd_file_path": "/path/to/CCD/file.cif",
  "seeds": [42, 123, 456]
}
```

**Required and Optional Fields**

- `queries` *(dict, required)* 
  - A dictionary containing one or more prediction targets. Each entry defines a single query (e.g., a protein or protein complex).
    - The keys (e.g., `query_1`, `query_2`, ...) uniquely identify each query and are used to name the corresponding output files.
    - If `n` queries are specified and `m` seeds are provided (see below), the model will perform `n × m` independent inference runs.

- `ccd_file_path` *(str, optional, default = None)*
  - Path to a [Chemical Component Dictionary (CCD)](https://www.wwpdb.org/data/ccd) mmCIF file containing definitions all ligands referenced across queries.
    - Standard ligands should be sourced from the official PDB CCD file linked above.
    - Custom ligands must be provided in MMCIF format, appended to the end of the CCD file. Identifiers of ligand definitions must exactly match the three-letter chemical component IDs used in the input queries.

- `seeds` *(list of int, optional, default = None)*
  - Specifies the exact random seeds to use for stochastic components of the inference process (e.g., dropout, sampling).
    - If provided, the model will run once per seed for each query.
    - Mutually exclusive with `num_seeds`.

- `num_seeds` *(int, optional, default = None)*
  - Specifies the number of random seeds to automatically generate.
    - Internally, seeds are sampled deterministically from a fixed global seed to ensure reproducibility.
    - Mutually exclusive with `seeds.`

If neither ```seeds``` nor ```num_seeds``` is provided, a single deterministic run will be performed per query using a default seed.


## 2. Queries
Each entry in the ```queries``` dictionary specifies a single bioassembly, which will be predicted in one forward pass of OpenFold3. To run **batch inference**, include multiple such query entries (e.g., ```query_1```, ```query_2```, ...) in the top-level ```queries``` field of the input JSON.

The key of each query (e.g., ```query_1```) is used to name output files or directories -- either by prefixing output files or creating a directory named after the key.

Each query entry is a dictionary with the following structure:

```
"query_1": {
  "chains": [ { ... }, { ... } ],
}
```

In the current inference release, the only required field is:
  - `chains` *(list of dict, required)*
    - A list of chain definitions, where each sub-dictionary specifies one chain in the assembly. See [Section 3](#3-chains) for a full breakdown of chain-level fields.


## 3. Chains

Each entry in the ```chains``` list defines one or more instances of a molecular chain in the bioassembly. The required and optional fields vary depending on the type of molecule (```protein```, ```rna```, ```dna```, or ```ligand```).

All chains must define a unique ```chain_ids``` field and appropriate sequence or structure information. Below are the supported molecule types and their associated schema:

  ### 3.1. Protein chains

  ```
  {
    "molecule_type": "protein",
    "chain_ids": "A",
    "sequence": "PVLSCGEWQCL",
    "use_msas": true,
    "use_main_msas": true,
    "use_paired_msas": true,
    "main_msa_file_paths": "/absolute/path/to/main_msa.sto/a3m",
    "paired_msa_file_paths": "/absolute/path/to/paired_msa.sto/a3m",
  }
  ```

  - `molecule_type` *(str, required)*
    - Must be "protein".

  - `chain_ids` *(str | list[str], required)*
    - One or more identifiers for this chain. Used to map sequences to structure outputs.

  - `sequence` *(str, required)*
    - Amino acid sequence (1-letter codes), supporting standard residues, X (unknown), and U (selenocysteine).

  - `non_canonical_residues` *(dict, optional, default = None)*
    - A dictionary mapping residue indices (1-based) to non-canonical residue names.
    - Note that MSA computation will only refer to the primary `sequence`.
    - Example: `{"1": "MHO", "5": "SEP"}`

  - `use_msas` *(bool, optional, default = true)*
    - Enables MSA usage. If false, empty MSA features are provided to the model. MSA-free inference mode is [discouraged](Inference.md#323--inference-without-msas) if the goal is to obtain the highest-accuracy structures.

  - `use_main_msas` *(bool, optional, default = true)*
    - Controls whether to use unpaired MSAs. 
    - For monomers or homomers, disabling this results in using only the single sequence(s) as MSA features.
    - For heteromers, disabling this results in using only the paired MSAs, including the query sequences, as MSA features.

  - `use_paired_msas` *(bool, optional, default = true)*
    - Controls the use of explicitly paired MSAs.
    - For homomers, main MSAs are internally concatenated and treated as implicitly paired, so disabling use_paired_msas does not change their MSA features.
    - For heteromers, paired alignments across chains are used if available and disabling use_paired_msas results in using only main MSAs as MSA features.

  - `main_msa_file_paths` *(str | list[str], optional, default = null)*
    - Path or list of paths to the MSA files for this chain.
    - Use this field only when running inference with **precomputed MSAs**. See the [Precomputed MSA documentation](precomputed_msas.md) for details.
    - If using the ColabFold MSA server (`--use_msa_server=True`), this field will be automatically populated and will **override any user-provided path**.

  - `paired_msa_file_paths` *(str | list[str], optional, default = null)*
    - Path or list of paths to paired MSA files for this chain, pre-paired in the context of the full complex.
    - Use this field only when running inference with **precomputed MSAs** and the corresponding query has at least two unique polymer chains. See the [Precomputed MSA documentation](precomputed_msas.md) for details.
    - If not provided, online MSA pairing can still be performed for protein chains if species information is available in one or more main MSA files per chain. See [Online MSA Pairing](precomputed_msas.md#6-online-msa-pairing-from-precomputed-msas) for details.
    - If using the ColabFold MSA server, this field is automatically populated and will **override any user-provided path**.


  ### 3.2. RNA Chains

  ```
  {
    "molecule_type": "rna",
    "chain_ids": "E",
    "sequence": "AGCU",
    "use_msas": true,
    "use_main_msas": true,
    "main_msa_file_paths": "/absolute/path/to/main_msa.sto/a3m",
  }
  ```

  - `molecule_type` *(str, required)*
    - Must be "rna".

  - `chain_ids` *(str | list[str], required)*
    - One or more identifiers for this chain. Used to map sequences to structure outputs.

  - `sequence` *(str, required)*
    - Nucleic acid sequence (1-letter codes).

  - `use_msas` *(bool, optional, default = true)*
    - Enables MSA usage. If false, a single-row MSA is constructed from the query sequence only.

  - `use_main_msas` *(bool, optional, default = true)*
    - Controls whether to use unpaired MSAs. For monomers or homomers, disabling this results in using only the single sequence.

  - `main_msa_file_paths` *(str | list[str], optional, default = null)*
    - Path or list of paths to the MSA files for this chain.
    - Use this field only when running inference with **precomputed MSAs**. See the [Precomputed MSA documentation](precomputed_msas.md) for details.


  ### 3.3. DNA Chains

  ```
  {
    "molecule_type": "dna",
    "chain_ids": "C",
    "sequence": "GACCTCT",
  }
  ```
  - `molecule_type` *(str, required)*
    - Must be "dna".

  - `chain_ids` *(str | list[str], required)*
    - One or more identifiers for this chain. Used to map sequences to structure outputs.

  - `sequence` *(str, required)*
    - Nucleic acid sequence (1-letter codes).


  ### 3.4. Small Molecule / Ligand Chains

  Ligand chains can be specified either using SMILES:
  ```
  {
    "molecule_type": "ligand",
    "chain_ids": "Z",
    "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
  }
  ```

  or using CCD codes:

  ```
  {
    "molecule_type": "ligand",
    "chain_ids": "I",
    "ccd_codes": "NAG",
  }
  ```
  - `molecule_type` *(str, required)*
    - Must be "ligand".

  - `chain_ids` *(str | list[str], required)*
    - Identifiers for the ligand chain(s).

  - `smiles` *(str, required if ccd_codes not given)*
    - Canonical SMILES string of the ligand.
    - Mutually exclusive with `ccd_codes`.

  - `ccd_codes` *(str | list[str], required if smiles not given)*
    - Three-letter CCD code for the ligand component. 
    - Support for providing a list of CCD codes (for instance for polymeric ligands) will be supported in a later release of the inference pipeline.
    - Mutually exclusive with `smiles`.

## 4. Example Input Json for a Single Query Complex

Below is a complete example of an input JSON file specifying a single bioassembly, consisting of:

- Two protein chains (`A` and `B`), with MSAs enabled

- One DNA chain (`C`)

- One RNA chain (`E`), with MSAs enabled

- Two types of non-covalently bound ligands:

  - A small molecule ligand (`Z`), defined by a SMILES string

  - A single-residue glycan-like ligand (`I`), specified CCD code `NAG`

```
{
    "seeds": [10, 42],
    "num_seeds": 2,
    "queries": {
        "query_1": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "PVLSCGEWQCL",
                    "use_msas": true,
                    "use_main_msas": true,
                    "use_paired_msas": true,
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "RPACQLWWSRGNWERINQLWW",
                    "use_msas": true,
                    "use_main_msas": true,
                    "use_paired_msas": true,
                },
                {
                    "molecule_type": "dna",
                    "chain_ids": "C",
                    "sequence": "GACCTCT",
                },
                {
                    "chain_ids": "E",
                    "molecule_type": "rna",
                    "sequence": "AGCU",
                    "use_msas": true,
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "Z",
                    "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "I",
                    "ccd_codes": ["NAG"],
                }
            ],
        }
    },
    "ccd_file_path": "/path/to/CCD/file.cif"
}
```

Additional example input JSON files can be found here:
- [Single-chain protein (monomer)](https://drive.google.com/file/d/1DtZN5jKIROVc_wd19wPWP7-lkYamInsV/view?usp=drive_link): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://drive.google.com/file/d/15zxPTsRYTrt_3rYfmjyC93QZUo2-M_YM/view?usp=drive_link): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (multimer)](https://drive.google.com/file/d/1tEakjCwaNbDAEzhnZgxToNFC1m-WmznL/view?usp=drive_link): Deoxy human hemoglobin (PDB: 1A3N)
- [Protein-ligand complex](https://drive.google.com/file/d/1HZHuyjBOJ9gMU99kbLG_W9aUJo4IGJ0g/view?usp=drive_link): Mcl-1 with small molecule inhibitor (PDB: 5FDR)