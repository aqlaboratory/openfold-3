# Understanding Precomputed MSA Handling

Here, we aim to provide additional explanations for the inner workings of the MSA components of the OF3 inference pipeline. If you need step-by-step instructions on how to generate MSAs using our OF3-style pipeline, refer to our [MSA Generation](msa_generation_how_to.md) document. If you need a guide on how to interface MSAs with the inference pipeline, go to the [Precomputed MSA How-To Guide](precomputed_msas_how_to.md).


## 1. MSASettings

This section will add some clarifying points about MSASettings.

The 3 main settings to update are:
1. *max_seq_counts*: A dictionary specifying how many sequences to read from each MSA file with the associated name. MSA files whose names are not provided in this dictionary *will not be parsed*. For example, if one wants `uniparc_hits.a3m` MSA files to be parsed, the following field should be specified:

```
dataset_config_kwargs:
  msa:
    max_seq_counts:  
      uniprot_hits: 50000
      mgnify_hits: 5000
      custom_database_hits: 10000
```

where up to the first 10000 sequences will be read from each `uniparc_hits.a3m` file.

2. *msas_to_pair*: The list of MSA filenames that contain species information that can be used for online pairing. See the [Online MSA Pairing](precomputed_msas.md#6-online-msa-pairing-from-precomputed-msas) section for details.

3. *aln_order*: The order in which to vertically concatenate MSA files for each chain for main MSA features. MSA files whose names are not provided in this list *will not be used*. For example, if one has MSA files named `mgnify_hits`, `uniprot_hits` and `uniparc_hits` and want to vertically concatenate them for each chain in this order, they should update the `runner.yml` as follows:

```
dataset_config_kwargs:
    aln_order:   
      - uniprot_hits
      - mgnify_hits
      - custom_database_hits
```

## 2. Online MSA pairing

This section will contain details on the workings/behavior of our online MSA pairing algorithm. Provided in the next internal release.

## 3. Chain Deduplication Utility

This section will contain details on how to use the chain representative logic of the MSA pipeline for highly redundant inference datasets, such as screens of a large number of small molecule ligands against the same protein chains or antibodies against the same antigen. Provided in the next internal release.

## 4. Preparsing Raw MSAs into NPZ Format