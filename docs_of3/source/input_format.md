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
  "msa_directory_path": "/path/to/alignment/dir",
  "seeds": [42, 123, 456]
}
```

*Required and Optional Fields*

```queries``` (dict, required)
A dictionary containing one or more prediction targets. Each entry defines a single query (e.g., a protein or protein complex). The keys (e.g., ```query_1```, ```query_2```, ...) uniquely identify each query and are used to name the corresponding output files.
    - For large-scale runs, keys can be automatically generated if omitted.
    - If `n` queries are specified and `m` seeds are provided (see below), the model will perform `n × m` independent inference runs.


```ccd_file_path``` (str, optional, default = null)
Path to a [Chemical Component Dictionary (CCD)]() mmCIF file containing definitions for any custom ligands used across queries.

    - All custom ligands for all queries must be included in a single CCD file.
    - Ligand definitions must match the three-letter chemical component IDs used in the input query definitions.


```msa_directory_path``` (str, optional, default = null)
Path to a directory containing precomputed multiple sequence alignments (MSAs).

- **Note**: This is reserved for future versions. The current implementation only supports online MSA generation via the MMseqs2 server.


```seeds``` (list of int, optional, default = null)
Specifies the exact random seeds to use for stochastic components of the inference process (e.g., dropout, sampling).
    - If provided, the model will run once per seed for each query.
    - Mutually exclusive with `num_seeds`.


```num_seeds``` (int, optional, default = null)
Alternative to `seeds`. Specifies the number of random seeds to automatically generate.
    - Internally, seeds are sampled deterministically from a fixed global seed to ensure reproducibility.
    - Mutually exclusive with seeds.

If neither ```seeds``` nor ```num_seeds``` is provided, a single deterministic run will be performed per query using a default seed.


## 2. Queries
Each entry in the ```queries``` dictionary specifies a single bioassembly, which is predicted in one forward pass of XXX. To run batch inference, define multiple such query entries (e.g., ```query_1```, ```query_2```, ...) in the top-level ```queries``` field of the input JSON.

The query key (e.g., ```query_1```) is used to name output files or directories (e.g., by prefixing output files or creating a directory named after the key — final behavior TBD).

Each query entry is a dictionary with the following structure:

```
"query_1": {
  "chains": [ { ... }, { ... } ],
  "covalent_bonds": [ [...], [...] ],
  "paired_msa_file_path": "path/to/paired_msa.a3m"
}
```

#### Fields

```chains``` (list of dict, required)
A list of chain definitions, where each sub-dictionary specifies one chain in the assembly.

See Section 2.1 for a full breakdown of chain-level fields.

```covalent_bonds``` (list of list, optional, default = null)
A list of atom pairs across chains between which covalent bonds should be enforced in the predicted structure.

Each sublist should define one covalent bond (e.g., ```[{"chain": "A", "residue": 42, "atom": "SG"}, {"chain": "B", "residue": 17, "atom": "SG"}]```).

Useful for modeling disulfide bridges or engineered cross-links.

See Section 2.2 for formatting details.

```paired_msa_file_path``` (str, optional, default = null)
Path to a precomputed paired MSA for the entire assembly.

Mutually exclusive with providing paired MSAs at the chain level (i.e., in individual chains entries).

When provided, overrides any paired MSAs defined per chain.


### 2.1. Chains

Each entry in the ```chains``` list defines one or more instances of a molecular chain in the bioassembly. The required and optional fields vary depending on the type of molecule (```protein```, ```rna```, ```dna```, or ```ligand```).

All chains must define a unique ```chain_ids``` field and appropriate sequence or structure information. Below are the supported molecule types and their associated schema:

#### Protein chains

{
  "molecule_type": "protein",
  "chain_ids": "A",
  "sequence": "PVLSCGEWQCL",
  "non_canonical_residues": { "5": "SEP" },
  "use_msas": true,
  "use_main_msas": true,
  "use_paired_msas": true,
  "main_msa_file_paths": "/abs/path/to/main.a3m",
  "paired_msa_file_paths": "/abs/path/to/paired.a3m",
  "templates": [ ... ],
  "use_templates": true
}

- ```molecule_type``` (str, required): Must be "protein".

- ```chain_ids``` (str | list[str], required): One or more identifiers for this chain. Used to map sequences to structure outputs.

- ```sequence``` (str, required): Amino acid sequence (1-letter codes), supporting standard residues, X (unknown), and U (selenocysteine).

- ```non_canonical_residues``` (dict, optional): Maps residue positions to CCD codes for non-canonical residues.

- ```use_msas``` (bool, optional, default = true): Enables MSA usage. If false, a single-row MSA is constructed from the query sequence only.

- ```use_main_msas``` (bool, optional, default = true): Controls whether to use unpaired MSAs.

    - For monomers or homomers, disabling this results in using only the single sequence.

- ```use_paired_msas``` (bool, optional, default = true): Controls use of explicitly paired MSAs.

    - For heteromers, paired alignments across chains are used if available.

    - For homomers, main MSAs are internally concatenated and treated as implicitly paired.

- ```main_msa_file_paths``` (str | list[str], optional): Path(s) to main MSAs (.sto, .a3m, or .npz).

- ```paired_msa_file_paths``` (str | list[str], optional): Path(s) to paired MSAs.

- ```use_templates``` (bool, optional, default = true): Enables use of structural templates.


#### RNA Chains

```
{
  "molecule_type": "rna",
  "chain_ids": "E",
  "sequence": "AGCU",
  "non_canonical_residues": { "1": "2MG", "4": "5MC" },
  "use_msas": true,
  "use_main_msas": true,
  "use_paired_msas": true,
  "main_msa_file_paths": "/abs/path/to/main.a3m",
  "paired_msa_file_paths": "/abs/path/to/paired.a3m"
}
```

- ```molecule_type``` (str, required): Must be "rna".

- ```chain_ids```, sequence, and non_canonical_residues: Same as for proteins.

- ```use_msas```, use_main_msas, use_paired_msas: Behave the same as for proteins.

- ```main_msa_file_paths```, paired_msa_file_paths: Path(s) to MSA files.


#### DNA Chains

{
  "molecule_type": "dna",
  "chain_ids": "C",
  "sequence": "GACCTCT",
  "non_canonical_residues": { "1": "6OG", "2": "6MA" }
}
```molecule_type``` (str, required): Must be "dna".

```chain_ids``` and ```sequence```: As above.

```non_canonical_residues``` (dict, optional): Maps positions to modified base codes.

```sdf_file_path (str, optional)```: Path to a structural definition of the DNA if needed for non-standard constructs.


#### Small Molecule / Ligand Chains

**Option 1: Using SMILES**

{
  "molecule_type": "ligand",
  "chain_ids": "Z",
  "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
}

**Option 2: Using CCD Codes**

{
  "molecule_type": "ligand",
  "chain_ids": "I",
  "ccd_codes": ["NAG", "FUC", "FUC"],
  "covalent_bonds": [
    [[1, "O4"], [2, "C1"]],
    [[2, "O4"], [3, "C1"]]
  ]
}
```molecule_type``` (str, required): Must be "ligand".

```chain_ids``` (str | list[str], required): Identifiers for the ligand chain(s).

```smiles``` (str, required if ccd_codes not given): Canonical SMILES string of the ligand.

```ccd_codes``` (str | list[str], required if smiles not given): One or more three-letter CCD codes for the ligand components. Used for glycans or multi-residue ligands.

```covalent_bond``` (list, optional): Only valid when using ccd_codes. Specifies covalent bonds between components, useful for branched polymers or glycans.

```sdf_file_path``` (str, optional): Optional SDF structure file for precise geometry specification.

Note: smiles and ccd_codes are mutually exclusive — exactly one must be provided.


### Complete Example Input Format