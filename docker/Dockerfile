# Full performance multi-stage build with complete CUDA toolchain
ARG CUDA_BASE_IMAGE_TAG=12.1.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_BASE_IMAGE_TAG} AS builder

# Environment mode: "lock" for reproducible builds, "yaml" for flexible dev builds
ARG BUILD_MODE=yaml

# Install complete build dependencies including CUDA compiler tools
RUN apt-get update && apt-get install -y \
    wget \
    libopenmpi-dev \
    libaio-dev \
    git \
    python3-dbg \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install miniforge
# FIXME this needs to be pinned, with more recent versions (25.11.0-1) the package resolution is stuck
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy environment files for both modes (small files, good for caching)
# To regenerate the lock file, see docker/DOCKER.md
# Use BUILD_MODE=yaml (default) for reproducible builds, BUILD_MODE=yml for flexible dev builds
COPY environments/production-linux-64.lock /opt/openfold3/production-linux-64.lock
COPY environments/production-linux-64.yml /opt/openfold3/production-linux-64.yml

# Install environment based on BUILD_MODE
# - lock: uses conda-lock for exact reproducible builds (training/production)
# - yaml: uses mamba env create for flexible version resolution (development/testing)
RUN mamba install -n base -c conda-forge conda-lock --yes \
    && if [ "$BUILD_MODE" = "lock" ]; then \
         conda-lock install --name openfold3 /opt/openfold3/production-linux-64.lock; \
       elif [ "$BUILD_MODE" = "yaml" ]; then \
         mamba env create -f /opt/openfold3/pproduction-linux-64 --name openfold3; \
       else \
         echo "Invalid BUILD_MODE: $BUILD_MODE. Use 'lock' or 'yaml'." && exit 1; \
       fi \
    && mamba clean --all --yes \
    && conda clean --all --yes

# Activate the openfold3 environment by default
ENV PATH=/opt/conda/envs/openfold3/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/openfold3
ENV CONDA_DEFAULT_ENV=openfold3

# Copy the minimal set of files needed to install the package
COPY pyproject.toml /opt/openfold3/
COPY openfold3/__init__.py /opt/openfold3/openfold3/
COPY scripts/ /opt/openfold3/scripts/

# Install third party dependencies
WORKDIR /opt/
RUN /opt/openfold3/scripts/install_third_party_dependencies.sh

# Install the package
WORKDIR /opt/openfold3
RUN python -m uv pip install --editable --no-deps .

# Set CUDA architecture for compilation (adjust based on your GPU)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# # Pre-compile DeepSpeed operations with full performance - run on GPU machines only
# RUN python3 -c "import deepspeed; deepspeed.ops.op_builder.EvoformerAttnBuilder().load()" || \
#     python3 -c "import deepspeed; print('DeepSpeed ops loaded successfully')"

# Devel stage - use devel image for full CUDA support
ARG CUDA_BASE_IMAGE_TAG=12.1.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_BASE_IMAGE_TAG} AS devel

# Install devel dependencies
RUN apt-get update && apt-get install -y \
    libopenmpi3 \
    libaio1 \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove only documentation and samples, keep compiler tools
RUN rm -rf /usr/local/cuda/doc \
    && rm -rf /usr/local/cuda/extras \
    && rm -rf /usr/local/cuda/samples \
    && rm -rf /usr/local/cuda/src \
    && rm -rf /usr/local/cuda/nsight* \
    && rm -rf /usr/local/cuda/lib64/libcudart_static.a \
    && rm -rf /usr/local/cuda/lib64/libcublas_static.a \
    && rm -rf /usr/local/cuda/lib64/libcurand_static.a \
    && rm -rf /usr/local/cuda/lib64/libcusolver_static.a \
    && rm -rf /usr/local/cuda/lib64/libcusparse_static.a \
    && rm -rf /usr/local/cuda/lib64/libnpp_static.a \
    && rm -rf /usr/local/cuda/lib64/libnvblas_static.a \
    && rm -rf /usr/local/cuda/lib64/libnvtoolsext_static.a \
    && rm -rf /usr/local/cuda/lib64/libnvrtc_static.a \
    && rm -rf /usr/local/cuda/lib64/libnvrtc-builtins_static.a

# Copy the entire conda environment
COPY --from=builder /opt/conda /opt/conda

# Copy CUTLASS
COPY --from=builder /opt/cutlass /opt/cutlass

# Activate the openfold3 environment by default
ENV PATH=/opt/conda/envs/openfold3/bin:/opt/conda/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/openfold3
ENV CONDA_DEFAULT_ENV=openfold3

# Ensure interactive shells also activate openfold3
RUN /opt/conda/bin/conda init bash \
    && echo "conda activate openfold3" >> /root/.bashrc

# Set environment variables
ENV CUTLASS_PATH=/opt/cutlass
ENV KMP_AFFINITY=none
ENV LIBRARY_PATH=/opt/conda/envs/openfold3/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/openfold3/lib:$LD_LIBRARY_PATH

# Copy the entire source tree directly (at the very end for optimal caching)
COPY . /opt/openfold3

WORKDIR /opt/openfold3

# Test stage - build on devel layer with test dependencies
FROM devel AS test

WORKDIR /opt/openfold3
# Only uv supports --dependency-groups from pyproject.toml
RUN uv pip install --dependency-groups=test
RUN uv pip install --no-deps --editable .
