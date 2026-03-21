#!/bin/bash

# -e (errexit): Exit immediately if any command returns a non-zero exit status
# -u (nounset): Exit if you try to use an uninitialized variable
# -o pipefail: Exit if any command in a pipeline fails (not just the last one)
set -euo pipefail

# These are necessary for subsequent (runtime) compilation of Deepspeed
echo "Download CUTLASS, required for Deepspeed Evoformer attention kernel"
git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1
conda env config vars set CUTLASS_PATH=$PWD/cutlass

# This setting is used to fix a worker assignment issue during data loading
conda env config vars set KMP_AFFINITY=none

# These will only be available outside of this script if it's sourced in the current shell
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/sbin  # TODO: Check if this is necessary, or is NERSC-specific
