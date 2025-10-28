# OpenFold3 Input Format

## 1. High-level Structure
The OpenFold3 inference pipeline takes a single JSON file as input, specifying the data and options required for structure prediction. This file can define multiple prediction targets (`queries`), which can be proteins, including individual protein chains and complexes, nucleic acids, and ligands. Multiple queries can be combined as follows: 

```json
{
  "queries": {
    "query_1": { ... },
    "query_2": { ... }
  },
}
```

**Required Input Fields**

- `queries` *(dict, required)* 
  - A dictionary containing one or more prediction targets. Each entry defines a single query (e.g., a protein or protein complex).
    - The keys (e.g., `query_1`, `query_2`, ...) uniquely identify each query and are used to name the corresponding output files.

**Reference Fields**
- `seeds` *(list, reference)*
  - This section is present in the logged `inference_query_set.json` that is part of the model run output as a reference of the seeds used.
  - However, this field should not be included as part of the input of an input `query.json`. Instead, the user should use either the `--num_model_seeds` command line argument or specify seeds in the `runner.yml`. See {ref}`Custom Model Seeds for more details <custom-random-seeds-inference>` 

(2-queries)=
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
    - A list of chain definitions, where each sub-dictionary specifies one chain in the assembly. See {ref}`Section 3 <3-chains>` for a full breakdown of chain-level fields.

(3-chains)=
## 3. Chains

Each entry in the ```chains``` list defines one or more instances of a molecular chain in the bioassembly. The required and optional fields vary depending on the type of molecule (```protein```, ```rna```, ```dna```, or ```ligand```).

All chains must define a unique ```chain_ids``` field and appropriate sequence or structure information. Below are the supported molecule types and their associated schema:

(31-protein-chains)=
  ### 3.1. Protein chains

  ```
  {
    "molecule_type": "protein",
    "chain_ids": "A",
    "sequence": "PVLSCGEWQCL",
    "use_msas": true,
    "use_main_msas": true,
    "use_paired_msas": true,
    "main_msa_file_paths": "/absolute/path/to/main_msas",
    "paired_msa_file_paths": "/absolute/path/to/paired_msas",
    "template_alignment_file_path": "/absolute/path/to/template_msa",
    "template_entry_chain_ids": ["entry1_A", "entry2_B", "entry3_A"],
  }
  ```

  - `molecule_type` *(str, required)*
    - Must be "protein".

  - `chain_ids` *(str | list[str], required)*
    - One or more identifiers for this chain. Used to map sequences to structure outputs.

  - `sequence` *(str, required)*
    - Amino acid sequence (1-letter codes), supporting standard residues, X (unknown), and U (selenocysteine).

  - `non_canonical_residues` *(dict, optional, default = null)*
    - A dictionary mapping residue indices (1-based) to non-canonical residue names.
    - Note that MSA computation will only refer to the primary `sequence`.
    - Example: `{"1": "MHO", "5": "SEP"}`

  - `use_msas` *(bool, optional, default = true)*
    - Enables MSA usage. If false, empty MSA features are provided to the model. We suggest running MSA-free inference mode via a dummy MSA with only the query sequence and completely omitting MSA inputs is {ref}`discouraged <323-inference-without-msas>` if the goal is to obtain the highest-accuracy structures.

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
    - Use this field only when running inference with **precomputed MSAs**. See the {doc}`Precomputed MSA documentation <precomputed_msa_how_to>` for details.
    - If using the ColabFold MSA server (`--use_msa_server=True`), this field will be automatically populated and will **override any user-provided path**.

  - `paired_msa_file_paths` *(str | list[str], optional, default = null)*
    - Path or list of paths to paired MSA files for this chain, pre-paired in the context of the full complex.
    - Use this field only when running inference with **precomputed MSAs** and the corresponding query has at least two unique polymer chains. See the {doc}`Precomputed MSA documentation <precomputed_msa_how_to>` for details.
    - If not provided, online MSA pairing can still be performed for protein chains if species information is available in one or more main MSA files per chain. See {ref}`Online Cross-Chain Pairing in OF3 <3-online-msa-pairing>` for details.
    - If using the ColabFold MSA server, this field is automatically populated and will **override any user-provided path**.

  - `template_alignment_file_path` *(str, optional, default = null)*
    - Path to template alignment file for this chain.
    - Use this field only when running inference with **precomputed alignments**. See the {doc}`Running with Templates Documentation <template_how_to>` for details.
    - If using the ColabFold MSA server, this field is automatically populated and will **override any user-provided path**.

  - `template_entry_chain_ids` *(str, optional, default = null)*
    - !!! Currently, only populated automatically !!!
    - A list of template PDB entry + chain IDs to use for this chain.
    - Use this field only when running inference with **precomputed alignments**. See the {doc}`Running with Templates Documentation <template_how_to>` for details.
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
    - Enables MSA usage. If false, empty MSA features are provided to the model. We suggest running MSA-free inference mode via a dummy MSA with only the query sequence and completely omitting MSA inputs is {ref}`discouraged <323-inference-without-msas>` if the goal is to obtain the highest-accuracy structures.

  - `use_main_msas` *(bool, optional, default = true)*
    - Controls whether to use unpaired MSAs. For monomers or homomers, disabling this results in using only the single sequence.

  - `main_msa_file_paths` *(str | list[str], optional, default = null)*
    - Path or list of paths to the MSA files for this chain.
    - Use this field only when running inference with **precomputed MSAs**. See the {doc}`Precomputed MSA documentation <precomputed_msa_how_to>` for details.


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

```json
{
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
- [Single-chain protein (monomer)](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs/query_ubiquitin.json): Ubiquitin (PDB: 1UBQ)
- [Multi-chain protein with identical chains (homomer)](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs/query_homomer.json): GCN4 leucine zipper (PDB: 2ZTA)
- [Multi-chain protein with different chains (multimer)](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs/query_multimer.json): Deoxy human hemoglobin (PDB: 1A3N)
- [Protein-ligand complex](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs/query_protein_ligand.json): Mcl-1 with small molecule inhibitor (PDB: 5FDR)
- [Multiple Protein-ligand complexes](https://github.com/aqlaboratory/openfold-3/tree/main/examples/example_inference_inputs/query_protein_ligand_multiple.json): Two queries with Mcl-1 and different small molecule inhibitors (PDB: 5FDR)