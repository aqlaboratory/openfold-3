# Overview

We use the workflow manager [snakemake]() to help orchestrate large scale MSA generation. Snakemake distributes jobs efficiently across single node or across a whole cluster. We used this approach to generate MSAs at scale for the PDB and monomer distillation sets.

# Usage

First, make sure to download our alignment databases using [this script](). Next, create a conda environment using the `aln_env.yml` file.  Next, modify `example_msa_config.json` so that the paths to databases and environments match your system. You can verify snakemake is configured by running a dryrun with snakemake: 

```
snakemake -np -s MSA_Snakefile --configfile <path/to/config.json>
```

If this runs successfully, launch the main job

```
snakemake -s MSA_Snakefile \
    --cores <available cores> \ 
    --configfile <path/to/config.json>  \
    --nolock  \ 
    --keep-going \
    --latency-wait 120
```

## Best practices

While snakemake has great support for running individual jobs across a cluster, we find that the optimal way to use out alignment pipeline on a typical academic HPC is to submit indpendent snakemake jobs that use a whole node at a time. The main reason for this is that alignments generally work best when the alignment databases are stored on node-local SSD based storage. This typically requires copying data each time a job is run on a node as in most clusters node-local storage is not peristent. Therefore a typical workflow involves first copying alignment DBs to a node, and then running snakemake locally on that node.


# Configuration settings 

An input configuration in json format must be passed, see `example_msa_config.json` for defaults. Description of fields

```
input_fasta: absolute or relative path to input fasta 
openfold_env: path to openfold conda environment, ie ~/miniforge3/envs/of3
databases: one or more of [uniref90, uniprot, mgnify, cfdb, bfd]
base_database_path: The base directory all alignments dbs are in. Should have the format `{base_directory}/{db}/{db}.fasta` for uniref90, uniprot, mgnify. cfdb/bfd must be downloaded and unpacked into `{base_directory}/{bfd|cfdb}/`
output_directory: output folder to write MSAs to 
jackhmmer_output_format: output format to write jackhmmer MSAs in, one of ["sto", a3m]
jackhmmer_threads: number of thread to use for jackhmmer
hhblits_threads: number of threads to use for hhblits
tmpdir : temporary directory to generate intermediate files
run_template_search: whether or not to run template search with hmmsearch. This requires either: uniref90 to be set as a database, or previously completed unirer90 alignments.
```
