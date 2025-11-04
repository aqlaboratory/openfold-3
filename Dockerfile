# Full performance multi-stage build with complete CUDA toolchain
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

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
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy and install dependencies with aggressive cleanup
COPY environments/production.yml /opt/openfold3/environment.yml
RUN mamba env update -n base --file /opt/openfold3/environment.yml --prune \
    && mamba clean --all --yes \
    && conda clean --all --yes

# Copy the entire source tree
COPY . /opt/openfold3/

# Install third party dependencies
WORKDIR /opt/
RUN /opt/openfold3/scripts/install_third_party_dependencies.sh

# Install the package
WORKDIR /opt/openfold3
RUN python3 setup.py install

# Set CUDA architecture for compilation (adjust based on your GPU)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# # Pre-compile DeepSpeed operations with full performance - run on GPU machines only
# RUN python3 -c "import deepspeed; deepspeed.ops.op_builder.EvoformerAttnBuilder().load()" || \
#     python3 -c "import deepspeed; print('DeepSpeed ops loaded successfully')"

# Runtime stage - use devel image for full CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS runtime

# Install runtime dependencies
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
ENV PATH=/opt/conda/bin:$PATH

# Copy CUTLASS
COPY --from=builder /opt/cutlass /opt/cutlass

# Set environment variables
ENV CUTLASS_PATH=/opt/cutlass
ENV KMP_AFFINITY=none
ENV LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Copy the entire source tree to the runtime image
COPY --from=builder /opt/openfold3 /opt/openfold3

WORKDIR /opt/openfold3
