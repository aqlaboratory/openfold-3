# OpenFold3 Setup 

(openfold3-installation)=
## Installation

### Pre-requisites

OpenFold3 inference requires a system with a GPU with a minimum of CUDA 12.1 and 32GB of memory. Most of our testing has been performed on A100s with 40GB of memory. 

It is also recommended to use [Mamba](https://mamba.readthedocs.io/en/latest/) to install some of the packages.


### Installation via pip and mamba (recommended) 

1. Create a fresh mamba environment with python. Python versions 3.10 - 3.13 are supported

```bash
mamba create -n of3-pip-Oct24-server python=3.13 
```

2. Install openfold3 the pypi server:

```bash
pip install openfold3
```

to install GPU accelerated {doc}`cuEquivariance attention kernels <kernels>`, use: 

```bash
pip install openfold3[cuequivariance]
```

3. Install `kalign2` using mamba.

```bash
mamba install kalign2 -c bioconda
```

### OpenFold3 Docker Image

A compressed version of the OpenFold3 image is available on HuggingFace at this [link](https://huggingface.co/OpenFold/OpenFold3/tree/main/docker_image) The size of the compressed image is ~14GB.

To verify the compressed file is not unintentionally corrupted, you can check against the md5 checksum file provided with the following command and expected output.

```bash
$ md5sum -c openfold3_image.tar.bz2.md5
openfold3_image.tar.bz2: OK
```

The image may then be unpacked with the following command:

```bash
docker load --input openfold3_image.tar.bz2
```


### Building the OpenFold3 Docker Image 

If you would like to build an OpenFold docker image locally, we provide a dockerfile. You may build this image with the following command:

```bash
docker build -f Dockerfile -t openfold-docker .
```

(openfold3-parameters)=
## Downloading OpenFold3 model parameters

On the first inference run, default model parameters will be downloaded to the `$HOME/.openfold3`. To customize your checkpoint download path, you use one of the following options:

### Using `setup_openfold` 

We provide a one-stop binary that sets up openfold and runs integration tests. This binary can be called with:

```bash
setup_openfold
```

This script will:
- Create an `$OPENFOLD_CACHE` environment [Optional, default: `~/.openfold3`]
- Setup a directory for OpenFold3 model parameters [default: `~/.openfold3`]
    - Writes the path to `$OPENFOLD_CACHE/ckpt_path` 
- Download the model parameters, if the parameter file does not already exist 
- Optionally runs an inference integration test on two samples, without MSA alignments (~5 min on A100)
    - N.B. To run the integration tests, `pytest` must be installed. 


### Downloading the model parameters manually

The model parameters (~2GB) for the trained OpenFold3 model can be downloaded from [our AWS RODA bucket](https://registry.opendata.aws/openfold/) with the following script:

```bash
./openfold3/scripts/download_openfold_params.sh
```

By default, these weights will be downloaded to `~/.openfold3/`. 
You can customize the download directory by providing your own download directory as follows.

```bash
./scripts/download_openfold_params.sh --download_dir=<target-dir>
```

### Setting OpenFold3 Cache environment variable
You can optionally set your OpenFold3 Cache path as an environment variable:

```
export OPENFOLD_CACHE=`/<custom-dir>/.openfold3/`
```

This can be used to provide some default paths for model parameters (see section below).

### TL;DR: Where does OpenFold3 look for model parameters? 

OpenFold3 looks for parameters in the following order:
1. Use `inference_ckpt_path` that the user provides either as a command line argument or in the `experiment_settings.inference_ckpt_path` section in `runner.yml`
2. If the `$OPENFOLD_CACHE` value is set, either in the `runner.yml` under `experiment_settings.cache_path`, `$OPENFOLD_CACHE/ckpt_root` will be used
    - If no `$OPENFOLD_CACHE/ckpt_root` file is set, will attempt to download the parameters to `$OPENFOLD_CACHE` (and write `ckpt_root` file storing the cache)
3. If no `$OPENFOLD_CACHE` value is set, attempts to download the parameters to `~/.openfold3`.


## Running OpenFold Tests

OpenFold tests require [`pytest`](https://docs.pytest.org/en/stable/index.html), which can be installed with:

```bash
mamba install pytest
```

Once installed, tests can be run using:

```bash
pytest openfold3/tests/
```

To run the inference verification tests, run:
```bash
pytest tests/ -m "inference_verification"
```

Note: To build deepspeed, it may be necessary to include the environment `$LD_LIBRARY_PATH` and `$LIBRARY_PATH`, which can be done via the following

```
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
