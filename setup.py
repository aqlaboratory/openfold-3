# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

from scripts.utils import get_nvidia_cc

version_dependent_macros = [
    "-DVERSION_GE_1_1",
    "-DVERSION_GE_1_3",
    "-DVERSION_GE_1_5",
]

extra_cuda_flags = [
    "-std=c++17",
    "-maxrregcount=50",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]




setup(
    name="openfold3",
    version="0.1.0",
    description="A PyTorch reimplementation of DeepMind's AlphaFold 2 & 3",
    author="OpenFold Team",
    author_email="jennifer.wei@omsf.io",
    license="Apache License, Version 2.0",
    url="https://github.com/aqlaboratory/openfold3",
    packages=find_packages(exclude=["tests", "scripts"]),
    include_package_data=True,
    package_data={
        "openfold3": [
            # "core/kernels/cuda/csrc/*",
            "projects/*/config/*.yml",
        ],
        # "": ["resources/stereo_chemical_props.txt"],
    },
    # ext_modules=modules,
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10,"
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
