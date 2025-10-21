# OpenFold3 Installation

OpenFold3 inference requires a system with a GPU with a minimum of CUDA 12.1. Most of our testing has been performed on A100s with 80GB of memory. Documentation to support lower memory settings will be added shortly.

## OpenFold3 Docker Image tarball

A compressed version of the OpenFold3 image is available in this [google drive folder](https://drive.google.com/drive/u/0/folders/1_sKQhFU2cIb6DPYV8g9QxU5znYVdCd4N). The size of the compressed image is ~14GB.

To verify the compressed file is not unintentionally corrupted, you can check against the md5 checksum file provided with the following command and expected output.

```bash
$ md5sum -c openfold3_image.tar.bz2.md5
openfold3_image.tar.bz2: OK
```

The image may then be unpacked with the following command:

```
docker load --input openfold3_image.tar.bz2
```


## Building the OpenFold3 Docker Image 

If you would like to build an OpenFold docker image locally, we provide a dockerfile. You may build this image with the following command:

```
docker build -f Dockerfile -t openfold-docker .
```


## Installation via mamba 

Alternative, you can manually set up the environment by following these steps:

1. Follow steps 1-4 of the [OpenFold2 installation guide](https://openfold.readthedocs.io/en/latest/Installation.html), **but** replace the `environment.yml` file with `environments/production.yml` from the [OpenFold3 repository](https://github.com/aqlaboratory/openfold3/blob/inference-dev/environments/production.yml).
2. In particular, in **step 2**, run:
```
$ mamba env create -n openfold_env -f environments/production.yml
```

**Note:** You’ll need to have mamba installed; see the [mamba documentation](https://mamba.readthedocs.io/en/latest/) if needed.


## Downloading OpenFold3 model parameters

### Easiest: Use `./scripts/setup_openfold3.sh` to download and setup default parameter paths

The [`setup_openfold3.sh`](../../scripts/setup_openfold3.sh) script sets up quick defaults for parameter paths and saves selected paths as default variables. 

In detail, this script will:
- Setup an `$OPENFOLD_CACHE` environment [Optional, default: `~/.openfold3`]
- Setup a directory for OpenFold3 model parameters [default: `~/.openfold3`]
    - Writes the path to `$OPENFOLD_CACHE/ckpt_path` 
- Download the model parameters, if the parameter file does not already exist 
- Runs an inference integration test on two samples, without MSA alignments (~5 min on A100)

We recommend running this script as a one-stop script to download parameters and verify your installation.

### Downloading the model parameters manually

The model parameters (~5GB) for the trained OpenFold3 model can be downloaded from [our AWS RODA bucket](https://registry.opendata.aws/openfold/) with the following script:

```
./scripts/download_openfold_params.sh
```

By default, these weights will be downloaded to `~/.openfold3/`. 
You can customize the download directory by providing your own download directory as follows.

```
./scripts/download_openfold_params.sh --download_dir=<target-dir>
```

### Setting OpenFold3 Cache environment variable
You can optionally set your OpenFold3 Cache path as an environment variable:

```
export OPENFOLD_CACHE=`/<custom-dir>/.openfold3/`
```

This can be used to provide some default paths for model parameters (see section below).

### Where does OpenFold3 look for model parameters? 

OpenFold3 looks for parameters in the following order:
1. Use `inference_ckpt_path` that the user provides either as a command line argument or in the `experiment_settings.inference_ckpt_path` section in `runner.yml`
2. If the `$OPENFOLD_CACHE` value is set, either in the `runner.yml` under `experiment_settings.cache_path`, `$OPENFOLD_CACHE/ckpt_root` will be used
    - If no `$OPENFOLD_CACHE/ckpt_root` file is set, will attempt to download the parameters to `$OPENFOLD_CACHE` (and write `ckpt_root` file storing the cache)
3. If no `$OPENFOLD_CACHE` value is set, attempts to download the parameters to `~/.openfold3`.


## Running OpenFold Tests

OpenFold tests require the additional packages listed in `environments/development.txt`

These packages can be installed via the following steps:

1. Activate your Openfold mamba environment, e.g.
```
$ mamba activate openfold_env
```

2. Use `pip` to install the development dependencies 
```
$ pip install environments/development.txt
```

To run the tests, you may use `pytest`, e.g.
```
$ pytest tests/
```

To run the inference verification tests, run:
```
$ pytest tests/ -m "inference_verification"
```

Note: To build deepspeed, it may be necessary to include the environment `$LD_LIBRARY_PATH` and `$LIBRARY_PATH`, which can be done via the following

```
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```