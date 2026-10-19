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

"""Tests for the atom array alignment primitives in `alignment.py`.

The structures are moved by known rigid motions, so the expected coordinates are
the untouched coordinates read from the mmCIF fixtures.
"""

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

import openfold3
from openfold3.core.data.primitives.structure.alignment import coalign_atom_arrays

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"

# 90 degree rotation about z, applied as x' = R x + t
ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TRANSLATION = np.array([10.0, -20.0, 30.0])


def _load(pdb_id: str) -> AtomArray:
    # label_asym_id chain IDs give each ligand its own chain, as in preprocessing
    return pdbx.get_structure(
        pdbx.CIFFile.read(MMCIFS_DIR / f"{pdb_id}.cif"),
        model=1,
        use_author_fields=False,
        extra_fields=["occupancy"],
    )


def _moved(atom_array: AtomArray) -> AtomArray:
    moved = atom_array.copy()
    moved.coord = atom_array.coord @ ROTATION.T + TRANSLATION
    return moved


def test_coalign_atom_arrays_moves_ligand_into_fixed_frame():
    """1pzp's FTA ligand rides along with its protein's pocket alignment."""
    structure = _load("1pzp")
    protein = structure[struc.filter_amino_acids(structure)]
    ligand = structure[structure.res_name == "FTA"]

    aligned = coalign_atom_arrays(
        fixed=protein,
        mobile=_moved(protein),
        comobile=_moved(ligand),
        alignment_mask_atom_names=["N", "CA", "C"],
        mobile_distance_atom_names=["N", "CA", "C"],
        distance_threshold=10.0,
    )

    np.testing.assert_allclose(aligned.coord, ligand.coord, atol=1e-3)
