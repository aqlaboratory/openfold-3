# OpenFold3 Installation

A Docker-based setup is coming soon (estimated week of June 16).

## Temporary Installation Instructions

In the meantime, you can manually set up the environment by following these steps:

1. Follow steps 1-4 of the [OpenFold2 installation guide](https://openfold.readthedocs.io/en/latest/Installation.html), **but** replace the `environment.yml` file with `environments/production.yml` from the [OpenFold3 repository](https://github.com/aqlaboratory/openfold3/blob/inference-dev/environments/production.yml).
2. In particular, in **step 2**, run:
```
$ mamba env create -n openfold_env -f environments/production.yml
```

**Note:** You’ll need to have mamba installed; see the mamba documentation if needed.


### Known Issue: rdkit Conflict

Due to a conflict between `pip` and `conda`, `rdkit=2025` may be installed incorrectly.

You can check with:
```
mamba list | grep rdkit
```

If you see something like:
```l
ibrdkit   2025.03.1     h84b0b3c_0     conda-forge
rdkit      2023.9.6      pypi_0         pypi
```
You’ll need to correct this by removing the pip version and installing the correct conda package:
```
pip uninstall rdkit
mamba install rdkit
```