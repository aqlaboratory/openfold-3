# Precomputed Multiple Sequence Alignments

OpenFold3 can utilize precomputed multiple sequence alignments (MSAs) for making predictions. Generating MSAs offline, separately from the model forward pass is the preferred practise for large batch jobs to avoid the expenditure of GPU resources for MSA creation, which is primarily a CPU-bound operation.

## 1. Generating OF3-Style Precomputed MSAs

Documentation and code on how to use our Snakemake pipeline to generate OF3-style precomputed protein MSAs will be provided in the next upcoming internal release. Our RNA MSA generation pipeline will be provided in a later release.

## 2. Precomputed MSA File Format

This section details the format of the MSA files provided by our Snakemake pipeline. MSAs generated with a different method should follow this format.

...details here...

## 3. Precomputed MSA Directory Structure and Specifying Paths in the Inference Query

...Expected directory structure and file naming...

...example here...

TODO: clean up 3.1.-3.3.

There are 3 different ways of passing alignment data to the inference pipeline. All three options are equivalent.

### 3.1. Direct File Paths

The direct paths for all alignments for each chain can be passed into the query.json. In this case, you would specify the main_msa_paths as follows:

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A/cfdb_uniref30.a3m",
                        "alignments/7cnw_A/mgnify_hits.sto",
                        "alignments/7cnw_A/uniprot_hits.sto",
                        "alignments/7cnw_A/uniref90_hits.sto"
                    ]
                },
                ...
            ],
	        "use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

Note that the chain id does not need to match the query / chain name; the directory name is not used for the alignments.

The individual file names contain the name of the source msa database used to construct the alignments. The names of these files do need to match the msa names provided to the inference dataset pipelines. We will see an example of how to provide these msa database names to the inference dataset by overwriting the values in the runner.json

### 3.2. Folder Containing Alignments per Chain

You may also pass in the directory containing the alignments relevant to the chain. in this case, the contents of the directory should still contain individual files that contain the msa database name.

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A/",
                    ]
                },
                ...
            ],
			"use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

### 3.3. Compressed NPZ File Containing Contents of All Alignment Files

It is also an option to pass in compressed npz files containing the alignment data above. This is the default way alignments from the ColabFold MSA server are stored.

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A.npz",
                    ]
                },
                ...
            ],
	        "use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

Documentation on how to create compressed npz files is forthcoming

## 4. Modifying MSA Settings for Custom Precomputed MSAs

TODO: clean up 4.

Specifying MSA databses in the Inference Dataset
The MSASettings are used to specify how the MSA data should be processed into features for the inference pipeline. For example, these settings can be used to tune the number of sequence alignments used from each database, or the number of rows to be used for each database

More details on the MSASettings are coming. Some high level notes on terminology:

Main alignments: These are the alignments for each chain from each database
Paired alignments: For protein multimer complexes with different chains, paired alignments are the combination of alignments across different chains. Multiple combination strategies are possible, but by default, the alignments are paired through concatenation.

How to overwrite the MSA Settings in runner.yml:

It is possible to pass in an update to the MSASettings to match your alignments using the dataset_config_kwargs section in the runner.yml

The example above used alignments with the key names: cfdb_uniref, mgnify_hits, uniprot_hits, uniref90_hits. To indicate that these are the msa databases to be parsed, we can add the following to our runner.yml

```
dataset_config_kwargs:
  msa:
    # specifies the number of alignments to use from each database 
    max_seq_counts:  
      uniref90_hits: 10000
      uniprot_hits: 50000
      cfdb_uniref30: 10000000
      mgnify_hits: 5000
	# specifies which alignments should be used to create the paired alignment 
    msas_to_pair: ["uniprot_hits", "uniprot"]
	# specifies the alignment order
    aln_order:   
      - uniref90_hits
      - cfdb_uniref30
      - mgnify_hits
      - concat_cfdb_uniref100_filtered
```

This runner yaml can be passed as a command line argument to our run_openfold.py command

```
python run_openfold.py predict \
--query_json query_precomputed_full_path.json \
--use_msa_server=False \
--inference_ckpt_path=of3_v14_79-32000_converted.ckpt.pt \
--output_dir=precomputed_prediction_output/ \
--runner_yaml=inference_precomputed.yml 
```

## 5. Chain Deduplication Utility

...details on representative logic...

## 6. Online MSA Pairing from Precomputed MSAs

...MSA format with species info...

...examples & vs colabfold pairing...