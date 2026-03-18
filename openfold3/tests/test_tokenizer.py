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

from pathlib import Path

import pytest

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.tokenization import (
    tokenize_atom_array,
)
from openfold3.core.data.primitives.structure.unresolved import (
    add_unresolved_atoms_within_residue,
)
from openfold3.tests.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path(__file__).parent / "test_data" / "tokenization"

paths = []
ids = ["1ema", "1pwc", "5seb", "5tdj", "6znc"]
for id in ids:
    paths.append(
        (
            TEST_DIR / "inputs" / f"{id}_raw_bonds_unfiltered.npz",
            TEST_DIR / "outputs" / f"{id}_tokenized_bonds_unfiltered.npz",
        )
    )


@pytest.mark.parametrize(
    "input_atom_array_path, precomputed_atom_array_path",
    paths,
    ids=ids,
)
def test_tokenizer_integration(
    input_atom_array_path: Path,
    precomputed_atom_array_path: Path,
):
    """Checks that the tokenizer adds the correct token annotations to the atom array.

    Args:
        input_atom_array_path (Path):
            Path to the input atom array that is to be tokenized.
        precomputed_atom_array_path (Path):
            Path to the precomputed atom array with the expected token annotations.
    """
    atom_array_in = read_atomarray_from_npz(input_atom_array_path)
    atom_array_out = read_atomarray_from_npz(precomputed_atom_array_path)

    tokenize_atom_array(atom_array_in)

    assert_atomarray_equal(atom_array_out, atom_array_in)


def test_tokenizer_unresolved_atoms_in_residues(biotite_ccd_wrapper):
    test_cif_path = Path(__file__).parent / "test_data" / "mmcifs" / "1kd8.cif"
    test_cif_file, test_structure = parse_mmcif(test_cif_path, expand_bioassembly=True)

    resolved_array = add_unresolved_atoms_within_residue(
        test_structure, test_cif_file.block, biotite_ccd_wrapper
    )
    # Verify that residue 3B contains unresolved atoms
    original_residue_3B = test_structure[
        (test_structure.res_id == 3) & (test_structure.chain_id == "B")
    ]
    resolved_residue_3B = resolved_array[
        (resolved_array.res_id == 3) & (resolved_array.chain_id == "B")
    ]

    assert len(resolved_residue_3B) > len(original_residue_3B), (
        "Resolved residue 3B should contain more atoms than the original residue."
    )

    # Tokenize the resolved residue
    tokenize_atom_array(resolved_residue_3B)

    assert all(resolved_residue_3B.token_id == 0), (
        "Expect all atoms in residue with unresolved atoms map to same token id"
    )
