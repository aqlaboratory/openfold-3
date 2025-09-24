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

### Running OpenFold Tests

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

Note: To build deepspeed, it may be necessary to include the environment `$LD_LIBRARY_PATH` and `$LIBRARY_PATH`, which can be done via the following

```
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
