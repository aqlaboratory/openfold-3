FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    libopenmpi-dev \
    libaio-dev \
    git \
    python3-dbg

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-Linux-x86_64.sh"
RUN bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda
RUN rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

COPY environments/production.yml /opt/openfold3/environment.yml
# installing into the base environment since the docker container wont do anything other than run openfold
RUN mamba env update -n base --file /opt/openfold3/environment.yml

# Manually update rdkit due to rdkit being downgraded from pip install pdbeccutils
RUN pip -y uninstall rdkit
RUN mamba install rdkit
RUN mamba clean --all

COPY openfold3 /opt/openfold3/openfold3
COPY setup.py /opt/openfold3/setup.py
COPY scripts /opt/openfold3/scripts
COPY run_openfold.py /opt/openfold3/

WORKDIR /opt/openfold3

RUN python3 setup.py install
