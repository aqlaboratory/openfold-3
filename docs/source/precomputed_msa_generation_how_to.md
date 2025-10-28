# A How-To Guide for OF3-Style Precomputed MSA Generation

We use the workflow manager [snakemake](https://snakemake.readthedocs.io/en/stable/) to help orchestrate large scale MSA generation. Snakemake distributes jobs efficiently across single node or across a whole cluster. We used this approach to generate MSAs at scale for the PDB and monomer distillation sets. Our pipeline supports both protein alignments and RNA alignments.

(1-msa-generation-usage)=
## 1. Usage

1. Create a [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment using the `aln_env.yml` file
2. Download our alignment databases using the [`download_of3_databases.py`](../../scripts/snakemake_msa/download_of3_databases.py) script 
    - By default, running `python download_of3_databases.py` will download the UniRef90, UniProt, MGnify, and PDB SEQRES databases for proteins - requires *330GB* of disk space
    - Additional databases can be downloaded by appending one of the download database flags to the script execution command:
        - `--download-bfd` downloads **BFD** (Deepmind) - requires an additional *2.3TB* of disk space
        - `--download-cfdb` downloads **ColabFold** database (OF3 and the Steinneger lab, intended to replace BFD)  - requires an additional *1.5TB* of disk space
        - `--download-rna-dbs` downloads **Rfam, RNACentral, Nucleotide Collection** (RNA alignments) - requires an additional *27GB* disk space
3. Modify the example [protein](../../scripts/snakemake_msa/example_msa_config_protein.json) or [RNA](../../scripts/snakemake_msa/example_msa_config_RNA.json) configs so that the paths to databases and environments match the downloaded databases on your system. A detailed description of the config fields is listed below: 

- `input_fasta` *(Path)*
    - Absolute or relative path to input fasta.
- `openfold_env` *(Path)*
    - Path to openfold mamba/conda environment, for example ~/miniforge3/envs/of3.
- `databases` *(list[str])*
    - List of database names to generate alignments for. One or more of [uniref90, uniprot, mgnify, cfdb, bfd].
- `base_database_path`*(Path)*
    - The base directory all alignments dbs are in. Should have the format `{base_directory}/{db}/{db}.fasta` for uniref90, uniprot, mgnify. cfdb/bfd must be downloaded and unpacked into `{base_directory}/{bfd|cfdb}/`.
- `output_directory` *(Path)*
    - Output folder to write MSAs to.
- `jackhmmer_output_format` *(str)*
    - Output format to write jackhmmer MSAs in, one of ["sto", "a3m"].
- `jackhmmer_threads` *(int)*
    - Number of threads to use for jackhmmer.
- `nhmmer_threads` *(int)*
    - Number of threads to use for nhmmer.
- `hhblits_threads` *(int)*
    - Number of threads to use for hhblits.
- `tmpdir` *(Path)*
    - Temporary directory to generate intermediate files.
- `run_template_search`
    - Whether or not to run template search with hmmsearch. This requires either: uniref90 to be set as a database, or previously completed uniref90 alignments.


4. Verify snakemake is configured correctly by running a dryrun with snakemake. If this runs successfully, you should see no error messages, and list of alignment jobs the pipeline will run.

```
snakemake -np -s MSA_Snakefile --configfile <path/to/config.json>
```

5. You can then run the main alignment workflow like this. 

```
snakemake -s MSA_Snakefile \
    --cores <available cores> \ 
    --configfile <path/to/config.json>  \
    --nolock  \ 
    --keep-going \
    --latency-wait 120
```

(2-msa-generation-output)=
## 2. Output

For each unique sequence, the pipeline generates a directory of MSAs, with filenames indicating which MSA comes from which database. For instance, for three unique protein chains queried against BFD, PDB SEQRES (for template alignments), MGnify, UniProt and UniRef90, you should see:

```
alignments/
├── example_chain_A/
│   ├── bfd_hits.a3m
│   ├── hmm_output.sto
│   ├── mgnify_hits.a3m
│   ├── uniprot_hits.a3m
│   └── uniref90_hits.a3m
├── example_chain_B/
│   ├── bfd_hits.a3m
│   ├── hmm_output.sto
│   ├── mgnify_hits.a3m
│   ├── uniprot_hits.a3m
│   └── uniref90_hits.a3m
└── example_chain_C/
    ├── bfd_hits.a3m
    ├── hmm_output.sto
    ├── mgnify_hits.a3m
    ├── uniprot_hits.a3m
    └── uniref90_hits.a3m
```

whereas for an RNA chain queried against RNA databases, you get:

```
alignments
└── example_chain_D/
    ├── nt_hits.a3m
    ├── rfam_hits.sto
    └── rnacentral_hits.a3m
```

(3-msa-generation-preparsing-msas)=
## 3. Preparsing MSAs into NPZ format

Optionally, you can preparse the raw alignment files generated using our snakemake pipeline into NPZ format, which we recommend for large datasets with a redundant set of sequences to increase MSA parsing speed in the OF3 data pipeline and reduce the storage costs of the MSAs. See the main How-To document {ref}`NPZ section <3-preparsing-raw-msas-into-npz-format>` for details.

(4-msa-generation-adding-msa-paths)=
## 4. Adding MSA Paths to the Inference Query Json

The {doc}`inference query json <input_format>` specifies the input into the model. You can tell the data pipeline which MSAs to use for which chain by adding the paths to the MSAs of the corresponding chain's field in the json file. For example, for a complex with one of each of the above three protein chains and one of the RNA chains, you can do the following:

<details>
<summary>Query json with MSA paths example ...</summary>
<pre><code>
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "GCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLG",
                    "main_msa_file_paths": "alignments/example_chain_A"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "MSELDQLRQEAEQLKNQIRDARKACADATLSQI",
                    "main_msa_file_paths": "alignments/example_chain_B"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "C",
                    "sequence": "MASNNTASIAQARKLVEQLKMEANIDRIKVSK",
                    "main_msa_file_paths": "alignments/example_chain_C"
                },
                {
                    "molecule_type": "rna",
                    "chain_ids": "D",
                    "sequence": "AUGCCGAUUCGAACU",
                    "main_msa_file_paths": "alignments/example_chain_D"
                },
            ],
        }
    }
}
</code></pre>
</details>

For additional notes on adding MSA paths to the inference query json, see the main {ref}`How-To document MSA Paths <4-specifying-paths-in-the-inference-query-file>` section.

(5-msa-generation-best-practices)=
## 5. Best Practices and Additional Notes

1. *Run proteins and RNA separately*: If you need to align both proteins and RNA, you have to make two separate configs and have two separate calls to the pipeline, one for each modality.

2. *Run with one database at a time*: Running multiple databases at once will work just fine, but for generating MSAs at scale (for instance, for 100 sequences), we recommend running only a single database at a time - this leads to faster runtimes as the database can be more aggressively cached in memory.

3. *Use whole-node jobs on HPC clusters*: While snakemake has great support for running individual jobs across a cluster, we find that the optimal way to use our alignment pipeline on a typical academic HPC is to submit independent snakemake jobs that use a whole node at a time. The main reason for this is that alignments generally work best when the alignment databases are stored on node-local SSD based storage. This typically requires copying data each time a job is run on a node as in most clusters node-local storage is not peristent. Therefore a typical workflow involves first copying alignment DBs to a node, and then running snakemake locally on that node.
