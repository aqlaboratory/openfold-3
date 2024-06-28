FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# metainformation
LABEL org.opencontainers.image.version = "0.1.0"
LABEL org.opencontainers.image.authors = "OpenFold Team"
LABEL org.opencontainers.image.source = "https://github.com/aqlaboratory/openfold3"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04"

RUN apt-get update && apt-get install -y wget

RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get install -y libxml2 cuda-minimal-build-12-1 libcusparse-dev-12-1 libcublas-dev-12-1 libcusolver-dev-12-1 \
    git openmpi-bin libopenmpi-dev && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

COPY environment.yml /opt/openfold3/environment.yml

# installing into the base environment since the docker container wont do anything other than run openfold
RUN mamba env update -n base --file /opt/openfold3/environment.yml && mamba clean --all
RUN export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

COPY openfold3 /opt/openfold3/openfold3
COPY scripts /opt/openfold3/scripts
COPY run_pretrained_openfold.py /opt/openfold3/run_pretrained_openfold.py
COPY train_openfold.py /opt/openfold3/train_openfold.py
COPY setup.py /opt/openfold3/setup.py
RUN wget -q -P /opt/openfold3/openfold3/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
WORKDIR /opt/openfold3
RUN python3 setup.py install
RUN python3 -m unittest "$@" 