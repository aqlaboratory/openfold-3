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

import numpy as np

from openfold3.core.data.resources.residues import (
    MoleculeType,
    map_str_array_to_idx_array,
)


def test_rna_msa_spelled_with_thymine_reads_t_as_u():
    # An RNA alignment from a DNA database that spells uracil as T
    t_msa = np.array([list("ACGT-"), list("TTGA-")])
    u_msa = np.array([list("ACGU-"), list("UUGA-")])
    t_idx = map_str_array_to_idx_array(t_msa, MoleculeType.RNA)
    u_idx = map_str_array_to_idx_array(u_msa, MoleculeType.RNA)
    np.testing.assert_array_equal(t_idx, u_idx)


def test_protein_msa_keeps_threonine():
    idx = map_str_array_to_idx_array(np.array([list("T")]), MoleculeType.PROTEIN)
    rna_idx = map_str_array_to_idx_array(np.array([list("U")]), MoleculeType.RNA)
    assert idx[0, 0] != rna_idx[0, 0]
