# Copyright 2025 AlQuraishi Laboratory
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

"""IO functions for reading and writing AtomArray objects."""

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.bonds import BondList


def write_atomarray_to_npz(
    atom_array: AtomArray, output_file: Path, compressed: bool = True
) -> None:
    """Write an AtomArray object to a NPZ file.

    Args:
        atom_array (AtomArray):
            The AtomArray object to be written.
        output_path (Path):
            The path to the output file.
        compressed (bool):
            Whether to write the resulting npz in compressed format. Defaults to True.
    """
    npz_dict = {}

    # Write atom coordinates
    npz_dict["coord"] = atom_array.coord

    # Write bondlist
    if atom_array.bonds is not None:
        npz_dict["bonds"] = atom_array.bonds.as_array()

    # Write all other annotations
    for annotation in atom_array.get_annotation_categories():
        npz_dict[annotation] = getattr(atom_array, annotation)

    if compressed:
        np.savez_compressed(output_file, **npz_dict)
    else:
        np.savez(output_file, **npz_dict)


def read_atomarray_from_npz(input_file: Path, allow_pickle=False) -> AtomArray:
    """Reads an AtomArray from npz format.

    Args:
        input_file (Path):
            .npz file containing annotation-values pairs which the AtomArray is created
            from.
        allow_pickle (bool):
            Whether to allow loading pickled objects. Can be necessary if the saved
            AtomArray contained object-type annotations. Defaults to False.

    Returns:
        AtomArray object
    """

    npz_obj = np.load(input_file, allow_pickle=allow_pickle)

    # Initialize empty atom array
    n_atoms = npz_obj["coord"].shape[0]

    atom_array = AtomArray(n_atoms)

    # Fill in all annotations
    for annotation, values in npz_obj.items():
        if annotation == "coord":
            atom_array.coord = npz_obj["coord"]
        elif annotation == "bonds":
            atom_array.bonds = BondList(n_atoms, npz_obj["bonds"])
        else:
            atom_array.set_annotation(annotation, values)

    return atom_array
