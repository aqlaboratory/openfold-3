# Understanding Precomputed MSA Handling

Here, we aim to provide additional explanations for the inner workings of the MSA components of the OF3 inference pipeline. If you need step-by-step instructions on how to generate MSA using our OF3-style pipeline, refer to our [MSA Generation](msa_generation_how_to.md) document. If you need a guide on how to then interface these MSAs with the inference pipeline, go to the [Precomputed MSA How-To Guide](precomputed_msas_how_to.md).

## 1. MSA pairing

This section will contain details on the workings/behavior of our online MSA pairing algorithm. Provided in the next internal release.

## 2. Chain Deduplication Utility

This section will contain details on how to use the chain representative logic of the MSA pipeline for highly redundant inference datasets, such as screens of a large number of small molecule ligands against the same protein chains or antibodies against the same antigen. Provided in the next internal release.