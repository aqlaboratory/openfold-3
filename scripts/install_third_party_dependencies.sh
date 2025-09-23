#!/bin/bash

python setup.py install

echo "Download CUTLASS, required for Deepspeed Evoformer attention kernel"
git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1
conda env config vars set CUTLASS_PATH=$PWD/cutlass

# This setting is used to fix a worker assignment issue during data loading
conda env config vars set KMP_AFFINITY=none

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/sbin  # TODO: Check if this is necessary, or is NERSC-specific
