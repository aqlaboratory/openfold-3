# OpenFold3-preview

```{figure} ../../imgs/predictions_combined_dark.png
:width: 900px
:align: center
:class: only-dark
:alt: Comparison of OpenFold and experimental structures on 5sgz (left), 3hfm (bottom right), 7ogs (top right)
```
```{figure} ../../imgs/predictions_combined_light.png
:width: 900px
:align: center
:class: only-light
:alt: Comparison of OpenFold and experimental structures on 5sgz (left), 3hfm (bottom right), 7ogs (top right)
```

Welcome to the Documentation for [OpenFold3-preview](https://github.com/aqlaboratory/openfold-3), a biological structure prediction model based on DeepMind's 
[AlphaFold3](https://github.com/deepmind/alphafold3). Developed under a fully open source (Apache 2) license.

## Quick-Start Guide

1. Install OpenFold3 using our pip package, {doc}`more details here <Installation>`

```bash
pip install openfold3 
mamba install kalign2 -c bioconda
```

2. Setup your installation of OpenFold3 and download parameters with our script:
```bash
setup_openfold
``` 
3. Make your first prediction with:
```bash
run_openfold --query_json=examples/example_inference_inputs/ubiquitin_query.json
```


## Features

OpenFold3 replicates the full set of input features described in the [*AlphaFold3*](https://www.nature.com/articles/s41586-024-07487-w) publication. 

A summary of the features supported include:
- Structure prediction of standard and non-canonical protein, RNA, and DNA chains, and small molecules
- Pipelines for generating MSAs using the [ColabFold server](https://github.com/sokrypton/ColabFold) or using JackHMMER / hhblits following the AlphaFold3 protocol
- {doc}`Structure templates <template_how_to>` for protein monomers
- Kernel acceleration through [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance) and [DeepSpeed4Science](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/) kernels - more details {doc}`here <kernels>`
- Support for {doc}`multi-query jobs <input_format>` with {ref}`distributed predictions across multiple GPUs <inference-run-on-multiple-gpus>`
- Custom settings for {ref}`memory constrained GPU resources <inference-low-memory-mode>`

and more features to come...


```{toctree}
:caption: Getting Started
:hidden:
:maxdepth: 2

Installation.md
kernels.md
```

```{toctree}
:caption: How To Guides
:hidden: 

Inference
precomputed_msa_generation_how_to
precomputed_msa_how_to
template_how_to
```

```{toctree}
:caption: Reference 
:hidden: 

input_format
configuration_reference
```

```{toctree}
:caption: Deep Dives 
:hidden: 

precomputed_msa_explanation
template_explanation
```