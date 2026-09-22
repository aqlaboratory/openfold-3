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

import biotite.structure as struc
import numpy as np
import pytest

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.featurization.structure import create_token_bonds
from openfold3.core.data.primitives.structure.tokenization import (
    FORCE_ATOMIZATION_ANNOTATION,
    tokenize_atom_array,
)
from openfold3.core.data.primitives.structure.unresolved import (
    add_unresolved_atoms_within_residue,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.tests.utils.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path(__file__).parent / "test_data" / "tokenization"


def _make_cross_chain_polymer_bond(
    residue_name: str,
    molecule_type: MoleculeType,
    atom_names: list[str],
    endpoint_names: tuple[str, str],
) -> tuple[struc.AtomArray, tuple[int, int]]:
    """Create two canonical residues joined by a backbone-like cross-chain bond."""
    atom_names_per_array = atom_names * 2
    n_atoms_per_residue = len(atom_names)
    n_atoms = len(atom_names_per_array)
    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = np.zeros((n_atoms, 3))
    atom_array.chain_id = np.repeat(["A", "B"], n_atoms_per_residue)
    atom_array.res_id = np.ones(n_atoms, dtype=int)
    atom_array.res_name = np.repeat(residue_name, n_atoms)
    atom_array.atom_name = atom_names_per_array
    atom_array.element = [name[0] for name in atom_names_per_array]
    atom_array.set_annotation("molecule_type_id", np.repeat(molecule_type, n_atoms))

    endpoint_indices = (
        atom_names.index(endpoint_names[0]),
        n_atoms_per_residue + atom_names.index(endpoint_names[1]),
    )
    atom_array.bonds = struc.BondList(
        n_atoms,
        np.asarray([(*endpoint_indices, struc.BondType.SINGLE)], dtype=np.uint32),
    )
    return atom_array, endpoint_indices


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


@pytest.mark.parametrize(
    "residue_name,molecule_type,atom_names,endpoint_names",
    [
        pytest.param(
            "ALA",
            MoleculeType.PROTEIN,
            ["N", "CA", "C", "O", "CB"],
            ("C", "N"),
            id="cross_chain_peptide_like_c_n",
        ),
        pytest.param(
            "DA",
            MoleculeType.DNA,
            ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'"],
            ("O3'", "P"),
            id="cross_chain_phosphodiester_like_o3_p",
        ),
    ],
)
def test_force_atomization_preserves_backbone_like_cross_chain_bond(
    residue_name,
    molecule_type,
    atom_names,
    endpoint_names,
):
    atom_array, endpoint_indices = _make_cross_chain_polymer_bond(
        residue_name=residue_name,
        molecule_type=molecule_type,
        atom_names=atom_names,
        endpoint_names=endpoint_names,
    )
    force_atomization = np.zeros(len(atom_array), dtype=bool)
    force_atomization[list(endpoint_indices)] = True
    atom_array.set_annotation(FORCE_ATOMIZATION_ANNOTATION, force_atomization)

    tokenize_atom_array(atom_array)

    assert np.all(atom_array.is_atomized)
    assert len(np.unique(atom_array.token_id)) == len(atom_array)
    assert FORCE_ATOMIZATION_ANNOTATION not in atom_array.get_annotation_categories()

    token_index = np.unique(atom_array.token_id)
    token_bonds = create_token_bonds(atom_array, token_index).numpy()
    endpoint_token_ids = atom_array.token_id[list(endpoint_indices)]
    assert endpoint_token_ids[0] != endpoint_token_ids[1]
    assert token_bonds[tuple(endpoint_token_ids)] == 1
    assert token_bonds[tuple(endpoint_token_ids[::-1])] == 1
    assert np.count_nonzero(token_bonds) == 2


def test_false_force_atomization_marker_preserves_unmarked_behavior():
    unmarked, _ = _make_cross_chain_polymer_bond(
        residue_name="ALA",
        molecule_type=MoleculeType.PROTEIN,
        atom_names=["N", "CA", "C", "O", "CB"],
        endpoint_names=("C", "N"),
    )
    false_marked = unmarked.copy()
    false_marked.set_annotation(
        FORCE_ATOMIZATION_ANNOTATION, np.zeros(len(false_marked), dtype=bool)
    )

    tokenize_atom_array(unmarked)
    tokenize_atom_array(false_marked)

    assert FORCE_ATOMIZATION_ANNOTATION not in (
        false_marked.get_annotation_categories()
    )
    assert_atomarray_equal(unmarked, false_marked)
    assert len(np.unique(unmarked.token_id)) == 2
    assert not np.any(unmarked.is_atomized)
    assert not np.any(
        create_token_bonds(unmarked, np.unique(unmarked.token_id)).numpy()
    )
