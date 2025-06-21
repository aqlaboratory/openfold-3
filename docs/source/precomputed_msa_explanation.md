# Understanding Precomputed MSA Handling

Here, we aim to provide additional explanations for the inner workings of the MSA components of the OF3 inference pipeline. If you need step-by-step instructions on how to generate MSAs using our OF3-style pipeline, refer to our [MSA Generation](msa_generation_how_to.md) document. If you need a guide on how to interface MSAs with the inference pipeline, go to the [Precomputed MSA How-To Guide](precomputed_msas_how_to.md).

## 1. Main vs Paired MSAs

The Openfold3 inference pipeline can create input MSA features for protein and RNA chains. We differentiate between two types of MSAs
1. **main MSAs** can be provided for protein

...


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

TODO: add uniprot sequence header explanation

This formatting requirement originates from our OF3 MSA generation pipeline running jackhmmer to search sequences against UniProt and us using UniProt MSAs for cross-chain pairing as per the AF3 SI. An example `sto` format is:

```
# STOCKHOLM 1.0

#=GS sp|P53859|CSL4_YEAST/1-292               DE [subseq from] Exosome complex component CSL4 OS=Saccharomyces cerevisiae (strain ATCC 204508 / S288c) OX=559292 GN=CSL4 PE=1 SV=1
#=GS tr|A6ZRL0|A6ZRL0_YEAS7/1-292             DE [subseq from] Conserved protein OS=Saccharomyces cerevisiae (strain YJM789) OX=307796 GN=CSL4 PE=4 SV=1
#=GS tr|C7GPC7|C7GPC7_YEAS2/1-292             DE [subseq from] Csl4p OS=Saccharomyces cerevisiae (strain JAY291) OX=574961 GN=CSL4 PE=4 SV=1
... rest of the metadata field ...

5k36_I                                           GDPHMACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
sp|P53859|CSL4_YEAST/1-292                       ----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
tr|A6ZRL0|A6ZRL0_YEAS7/1-292                     ----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
tr|C7GPC7|C7GPC7_YEAS2/1-292                     ----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
... rest of the alignments ...
```

and an example `a3m` format is:

```
>5k36_I
GDPHMACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
>sp|P53859|CSL4_YEAST/1-292
----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
>tr|A6ZRL0|A6ZRL0_YEAS7/1-292
----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
>tr|C7GPC7|C7GPC7_YEAS2/1-292
----MACNFQFPEIAYPGKLICPQY--G---------T--E-NK-D-G-------E-D--IIFNYVPGPGTKL----IQ---Y----E--------H---N--G---RT-------------LEAITATL-VGTV-RC---E--E----E----K--KT-DQ-E--E---E---R--EGT-D----Q-S-T--E--E-E-
... rest of the alignments ...
```

where the first sequence is the query sequence and headers `sp|P53859|CSL4_YEAST/1-292`, `tr|A6ZRL0|A6ZRL0_YEAS7/1-292` and `tr|C7GPC7|C7GPC7_YEAS2/1-292` are parsed to get species IDs `YEAST`, `YEAS7` and `YEAS2` for the three aligned sequences.

## 3. Chain Deduplication Utility

This section will contain details on how to use the chain representative logic of the MSA pipeline for highly redundant inference datasets, such as screens of a large number of small molecule ligands against the same protein chains or antibodies against the same antigen. Provided in the next internal release.

## 4. Preparsing Raw MSAs into NPZ Format

Two of the main challenges with MSAs are 
- slow parsing of MSA `sto` or `a3m` files into a numpy array for further downstream processing
- large storage costs associated with MSA files

Preparsing raw MSA files into `npz` files addresses these issues by 
- moving the per-example numpy array conversion step into an offline preprocessing step that happens only once for each unique MSA
- saving the MSA arrays in a compressed format

We found this step to be necessary during training to avoid the online data processing pipeline to bottleneck the model forward/backward passes and to reduce the storage costs associated with our distillation set MSAs.

For inference, preparsing MSAs in to `npz` files can be useful when running large batch jobs on highly redundant datasets, for example when screening one or a few target protein against a library of small molecule ligands or antibodies. 