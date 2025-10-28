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
run_openfold --query_json=examples/
```


## Features

OpenFold3 replicates the full set of input features described in the [*AlphaFold 3*](https://www.nature.com/articles/s41586-024-07487-w) publication. 

A summary of the features supported include:
- Structure prediction of protein, small molecule, RNA, and DNA. Includes support for non-canonical residues.
- A pipeline for generating MSAs using the [ColabFold server](https://github.com/sokrypton/ColabFold) or using JackHMMER / hhblits following the AlphaFold3 protocol.
- Use {doc}`templates for structure predictions <template_how_to>`
- Support for using GPU accelerated [CuEquivariance kernels](https://docs.nvidia.com/cuda/cuequivariance)
- Support for {ref}`distributed predictions across multiple GPUs <inference-run-on-multiple-gpus>`
- Custom setting for {ref}`memory constrained GPU resources <inference-low-memory-mode>`

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