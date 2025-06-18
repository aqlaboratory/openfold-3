# OpenFold3 Installation

## Docker Setup

We provide a dockerfile as a recipe to build your own openfold3 environment. You may build your own docker image with the following command:
```
docker build -f Dockerfile -t openfold-docker .
```


## Alternative Temporary Installation Instructions

Alternative, you can manually set up the environment by following these steps:

1. Follow steps 1-4 of the [OpenFold2 installation guide](https://openfold.readthedocs.io/en/latest/Installation.html), **but** replace the `environment.yml` file with `environments/production.yml` from the [OpenFold3 repository](https://github.com/aqlaboratory/openfold3/blob/inference-dev/environments/production.yml).
2. In particular, in **step 2**, run:
```
$ mamba env create -n openfold_env -f environments/production.yml
```

**Note:** You’ll need to have mamba installed; see the mamba documentation if needed.


### Known Issue: rdkit Conflict

Due to a conflict between `pip` dependencies and `conda` dependencies, `rdkit=2025` may be installed incorrectly.

You can check with:
```
mamba list | grep rdkit
```

If you see something like:
```
librdkit   2025.03.1     h84b0b3c_0     conda-forge
rdkit      2023.9.6      pypi_0         pypi
```
You’ll need to correct this by removing the pip version and installing the correct conda package:
```
pip uninstall rdkit
mamba install rdkit
```