# Build Instructions for Dockerfile.blackwell

## Basic build command:

```bash
docker build -f Dockerfile.blackwell -t openfold-3-blackwell:latest .
```

This will create a Docker image named `openfold-3-blackwell` with the `latest` tag.


## test Pytorch and CUDA

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 openfold-3-blackwell:latest   python -c "import torch; print('CUDA:', torch.version.cuda); print('PyTorch:', torch.__version__)"  
```

Should print something like:
CUDA: 12.8
PyTorch: 2.7.0a0+ecf3bae40a.nv25.02


## test run_openfold inference example

docker run --gpus all -it --ipc=host --ulimit memlock=-1 \
    -v $(pwd):/output \
    -w /output openfold-3-blackwell:latest \
    run_openfold predict \
    --query_json=/opt/openfold-3/examples/example_inference_inputs/query_ubiquitin.json \
    --num_diffusion_samples=1 \
    --num_model_seeds=1 \
    --use_templates=false 