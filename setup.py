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

from setuptools import find_packages, setup

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
            "projects/*/config/*.yml",
        ],
        "": ["resources/stereo_chemical_props.txt"],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10,"
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["run_openfold=run_openfold:cli"]},
)
