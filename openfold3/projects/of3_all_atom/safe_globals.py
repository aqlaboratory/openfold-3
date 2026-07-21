# Copyright 2026 AlQuraishi Laboratory
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

"""Importing this module registers the globals referenced in of3_all_atom
checkpoints as safe for `torch.load(..., weights_only=True)`.
"""

import operator
from pathlib import PosixPath

import ml_collections as mlc
import torch

from openfold3.projects.of3_all_atom.model import OpenFold3

# PosixPath is registered under both keys since checkpoints saved on Python <=3.12
# pickle it as "pathlib.PosixPath", while Python >=3.13 pickles it as
# "pathlib._local.PosixPath" (pathlib's internal module structure changed, but
# PosixPath.__module__ still resolves to "pathlib" either way).
torch.serialization.add_safe_globals(
    [
        OpenFold3,
        mlc.ConfigDict,
        mlc.FieldReference,
        int,
        bool,
        float,
        operator.add,
        mlc.config_dict._Op,
        PosixPath,
        (PosixPath, "pathlib._local.PosixPath"),
    ]
)
