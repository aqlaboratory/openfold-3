<picture>
  <source media="(prefers-color-scheme: dark)" srcset="imgs/predictions_combined_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="imgs/predictions_combined_light.png">
  <img alt="Comparison of OpenFold and experimental structures" src="imgs/predictions_combined_light.png">
</picture>

# OpenFold3-preview
OpenFold3 is a biomolecular structure prediction model aiming to be a bitwise reproduction of DeepMind's 
[AlphaFold3](https://github.com/deepmind/alphafold3), developed by AQLab and the OpenFold consortium. This research preview is intended to gather community feedback and allow developers to start building on top of the OpenFold ecosystem. The OpenFold project is committed to long-term maintenance and open source support, and our repository is freely available for academic and commercial use under the Apache 2.0 license.

For our reproduction of AlphaFold2, please refer to the original [OpenFold repository](https://github.com/aqlaboratory/openfold).

## Features

OpenFold3 replicates the input features described in the [*AlphaFold3*](https://www.nature.com/articles/s41586-024-07487-w) publication, as well as batch job support and efficient inference.

A summary of our supported features includes:
- Structure prediction of standard and non-canonical protein, small molecule, RNA, and DNA chains
- A pipeline for generating MSAs using the [ColabFold server](https://github.com/sokrypton/ColabFold) or using JackHMMER / hhblits following the AlphaFold3 protocol
- Structure templates
- Kernel acceleration through [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance) and [DeepSpeed4Science](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/) - more details here (TODO: Link to kernels page on documentation)
- Support for multi-query jobs with automatic device parallelization (TODO: Add refs)

## Quick-Start for Inference

Make your first predictions with OpenFold3-preview in a few easy steps:

1. Install OpenFold3
```bash
pip install openfold3
```

2. Install `kalign2`, either [from source](https://msa.sbc.su.se/cgi-bin/msa.cgi) or through `mamba`:

```bash
mamba install bioconda::kalign2
```

3. Run your first prediction using the ColabFold MSA server

```
python run_openfold.py predict --query_json=examples/example_inference_inputs/ubiquitin_query.json
```

More information on how to customize your inference prediction can be found at our documentation home at https://openfold3.readthedocs.io/en/latest/. More examples for inputs and outputs can be found at (TODO: Add hugging face examples directory here)

## Benchmarking
OpenFold3-preview performs competitively with the state of the art in open source protein structure prediction, while being the only model to match AlphaFold3 on monomeric RNA structures.

TODO: Add benchmarking results here

## Documentation

Please visit our full documentation at https://openfold3.readthedocs.io/en/latest/

## Upcoming
The final OpenFold3 model is still in development, and we are actively working on the following features:
- Improved performance on par with AlphaFold3
- Training documentation & dataset release
- Workflows for training on custom non-PDB data
- Binding affinity prediction

## Contributing

If you encounter problems using OpenFold3, feel free to create an issue! We also
welcome pull requests from the community.

## Citing this Work

Please cite our technical report:

TODO: Include citation to whitepaper

Any work that cites OpenFold should also cite [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w).
