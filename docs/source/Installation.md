# OpenFold3 Setup 

(openfold3-installation)=
## Installation

### Pre-requisites

OpenFold3 inference requires a system with a GPU with a minimum of CUDA 12.1 and 32GB of memory. Most of our testing has been performed on A100s with 40GB of memory. 

It is also recommended to use [Mamba](https://mamba.readthedocs.io/en/latest/) to install some of the packages.


### Installation via pip and mamba (recommended) 

0. [Optional] Create a fresh mamba environment with python. Python versions 3.10 - 3.13 are supported

```bash
mamba create -n openfold3 python=3.13 
```

1. Install openfold3 the pypi server:

```bash
pip install openfold3
```

to install GPU accelerated {doc}`cuEquivariance attention kernels <kernels>`, use: 

```bash
pip install openfold3[cuequivariance]
```

To use AMD ROCm-compatible Triton kernels, first install the ROCm PyTorch wheel (which bundles ROCm Triton), then install openfold3:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install openfold3
```

After installation, verify your ROCm environment is correctly configured:

```bash
validate-openfold3-rocm
```

(installation-environment-variables)=
### Environment variables

OpenFold may need a few environment variables set so CUDA, compilation, and JIT-built extensions can be found correctly. 

- `CUDA_HOME` should point to the CUDA installation. On many HPC clusters you will this can be set by loading the appropriate toolchain using environment modules, for example `module load cuda`.  If you do not set this you will likely get a `No such file or directory: '/usr/local/cuda/bin/nvcc'` error. 
- `CUTLASS_PATH` will need to be set for most systems. If you do not set this you will get Deepspeed related errors such as `Error: Unable to JIT load the evoformer_attn op`. Generally this can be set using 
    ```bash
    # Start your environment which as openfold3 installed
    source .venv/bin/activate
    # Set CUTLASS_PATH using the resolved path  
    export CUTLASS_PATH=$(python - << 'PY'
    import cutlass_library, pathlib
    print(pathlib.Path(cutlass_library.__file__).resolve().parent.joinpath("source"))
    PY
    )
    ```
- `LD_LIBRARY_PATH` may need to be set to the matching CUDA directories. How to set this will depend on the system. 
    - Example: `export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"`
    - You can often run `find "$CUDA_HOME" -name 'libcurand.so*' 2>/dev/null` to find the CUDA layout of your system. 

- If you get a `/usr/bin/ld: cannot find -lcurand` error, this usually means the CUDA math libraries (which include `libcurand`) are not on your library search path. You may need to add the appropriate CUDA library directory to  `LIBRARY_PATH`. 
    - Example: `export LIBRARY_PATH="$(echo "$CUDA_HOME" | sed 's|/cuda/|/math_libs/|')/targets/sbsa-linux/lib:${LIBRARY_PATH:-}"`



### OpenFold3 Docker Image

#### Dockerhub

The OpenFold3 Docker Image is now available on Docker Hub: [openfoldconsortium/openfold3](https://hub.docker.com/repository/docker/openfoldconsortium/openfold3/general)

To get the latest stable version, you can use the following command

```bash
docker pull openfoldconsortium/openfold3:stable
```

#### GitHub Container Registry (GHCR)

You can download the openfold3 docker image from GHCR, you'll need to install 'gh-cli' first, instructions [here](https://github.com/cli/cli/blob/trunk/docs/install_linux.md). 

You'll need to authenticate with GitHub, make sure you request the `read:packages` scope. 

```bash
gh auth login --scopes read:packages
```

Verify that login succeeded and scope is assigned 

```bash
gh auth status 
github.com
  ✓ Logged in to github.com account ******* (/home/ubuntu/.config/gh/hosts.yml)
  - Active account: true
  - Git operations protocol: ssh
  - Token: gho_************************************
  - Token scopes: 'admin:public_key', 'gist', 'read:org', 'read:packages', 'repo'
```

Let's inject the GitHub token into the docker config. Note this will expire. 

```bash
gh auth token | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin
```

Pull the image itself 

```bash
docker pull ghcr.io/aqlaboratory/openfold-3/openfold3-docker:0.4.0
```

### Building the OpenFold3 Docker Image 

If you would like to build an OpenFold docker image locally, we provide a dockerfile. You may build this image with the following command:

```bash
docker build -f Dockerfile -t openfold-docker .
```

(setup-openfold3-parameters)=
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
    - Writes the path to `$OPENFOLD_CACHE/ckpt_root` 
- Download the model parameters, if the parameter file does not already exist. You will have the option to download one set of parameters or all parameters. See {doc}`parameters_reference` for more information on available parameters. 
- Download and setup the [Chemical Component Dictionary (CCD)](https://www.wwpdb.org/data/ccd) with [Biotite](https://www.biotite-python.org/latest/apidoc/biotite.structure.info.get_ccd.html)
- Optionally run an inference integration test on two samples, without MSA alignments (~5 min on A100)
    - N.B. To run the integration tests, `pytest` must be installed. 


**Downloading the model parameters manually**

If preferred, the model parameters (~2GB) for the trained OpenFold3 model can be downloaded from [our AWS RODA bucket](https://registry.opendata.aws/openfold/) using the AWS CLI as follows:

```bash
aws s3 cp s3://openfold/staging/of3-p2-155k.pt <dst_path> --no-sign-request
```

To use these checkpoints with OpenFold3, it is then necessary to pass in the full path to the parameters through the command line arguments, e.g. `--inference_ckpt_path`. See {ref}`Inference instructions <default-inference>` for more details. 


### Setting OpenFold3 Cache environment variable
You can optionally set your OpenFold3 Cache path as an environment variable:

```
export OPENFOLD_CACHE=`/<custom-dir>/.openfold3/`
```

This can be used to provide some default paths for model parameters (see section below).

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
