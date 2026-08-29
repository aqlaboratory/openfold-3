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

"""Tests for create_msa_feature_precursor_of3.

These specifically cover msa_mask correctness when chains in the same assembly
have unequal MSA depths and therefore require per-chain row padding. See
vstack_pad_msa_arrays and map_msas_to_tokens: the padding mask computed by
MsaArray.pad must be propagated all the way into MsaFeaturePrecursorOF3.msa_mask,
or the fabricated all-gap padding rows added to shallower chains get silently
reported as valid MSA rows.
"""

import numpy as np
from biotite.structure import Atom
from biotite.structure import array as biotite_array

from openfold3.core.data.primitives.featurization.msa import (
    create_msa_feature_precursor_of3,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    MsaRowCounts,
)
from openfold3.core.data.primitives.structure.tokenization import add_token_positions
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
)

N_RES_PER_CHAIN = 2


def _msa(n_rows: int, n_cols: int = N_RES_PER_CHAIN) -> MsaArray:
    return MsaArray(
        msa=np.full((n_rows, n_cols), "A"),
        deletion_matrix=np.zeros((n_rows, n_cols)),
    )


def _build_two_chain_atom_array():
    """Two chains ('A', 'B'), N_RES_PER_CHAIN single-atom residues each."""
    atoms = []
    for entity_id, chain_id in enumerate(["A", "B"]):
        for res_id in range(1, N_RES_PER_CHAIN + 1):
            atoms.append(
                Atom(
                    coord=[0.0, 0.0, 0.0],
                    chain_id=chain_id,
                    res_id=res_id,
                    res_name="ALA",
                    atom_name="CA",
                    element="C",
                    entity_id=entity_id,
                )
            )
    atom_array = biotite_array(atoms)
    atom_array.set_annotation("token_id", np.arange(len(atom_array)))
    add_token_positions(atom_array)
    return atom_array


def _build_msa_array_collection(
    chain_a_main_rows: int, chain_b_main_rows: int
) -> MsaArrayCollection:
    """Two-chain collection with independently-sized per-chain main MSAs.

    Both chains always contribute a query row; n_rows_total is the max total
    depth across chains, matching how the real pipeline pads shallower chains.
    """
    chain_a_total = 1 + chain_a_main_rows
    chain_b_total = 1 + chain_b_main_rows
    profile = np.zeros((N_RES_PER_CHAIN, len(STANDARD_RESIDUES_WITH_GAP_1)))
    return MsaArrayCollection(
        chain_id_to_rep_id={"A": "A", "B": "B"},
        chain_id_to_mol_type={"A": MoleculeType.PROTEIN, "B": MoleculeType.PROTEIN},
        rep_id_to_chain_id={"A": "A", "B": "B"},
        rep_id_to_mol_type={"A": MoleculeType.PROTEIN, "B": MoleculeType.PROTEIN},
        chain_id_to_query_seq={"A": _msa(1), "B": _msa(1)},
        chain_id_to_main_msa={
            "A": _msa(chain_a_main_rows),
            "B": _msa(chain_b_main_rows),
        },
        chain_id_to_profile={"A": profile.copy(), "B": profile.copy()},
        chain_id_to_deletion_mean={
            "A": np.zeros(N_RES_PER_CHAIN),
            "B": np.zeros(N_RES_PER_CHAIN),
        },
        row_counts=MsaRowCounts(
            n_rows_total=max(chain_a_total, chain_b_total),
            n_rows_paired_subsampled=0,
            n_rows_main_subsampled={"A": chain_a_main_rows, "B": chain_b_main_rows},
        ),
    )


def test_msa_mask_zeroed_for_padded_rows_of_shallower_chain():
    """Regression test: rows added to pad a shallower chain up to the assembly's
    max MSA depth must be masked as invalid (0), not as valid (1).

    Chain A has a deep MSA (query + 2 main = 3 rows), chain B is shallow (query
    only = 1 row). Chain B gets bottom-padded with 2 fabricated all-gap rows to
    reach the assembly-wide depth of 3. Those 2 rows must read 0 in msa_mask for
    chain B's own token columns.
    """
    atom_array = _build_two_chain_atom_array()
    collection = _build_msa_array_collection(chain_a_main_rows=2, chain_b_main_rows=0)

    precursor = create_msa_feature_precursor_of3(
        atom_array, collection, n_tokens=2 * N_RES_PER_CHAIN
    )

    # Token columns: chain A occupies [0, 1], chain B occupies [2, 3].
    chain_a_cols, chain_b_cols = slice(0, 2), slice(2, 4)

    # Chain A has no padding: every row is real, mask is all-valid.
    np.testing.assert_array_equal(
        precursor.msa_mask[:, chain_a_cols], np.ones((3, N_RES_PER_CHAIN))
    )

    # Chain B: only row 0 (the query row) is real; rows 1-2 are fabricated
    # padding and must be masked out.
    expected_chain_b_mask = np.array([[1, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(
        precursor.msa_mask[:, chain_b_cols], expected_chain_b_mask
    )


def test_msa_mask_all_valid_when_chain_depths_are_equal():
    """Sanity check: when no chain needs padding, msa_mask stays all-valid."""
    atom_array = _build_two_chain_atom_array()
    collection = _build_msa_array_collection(chain_a_main_rows=2, chain_b_main_rows=2)

    precursor = create_msa_feature_precursor_of3(
        atom_array, collection, n_tokens=2 * N_RES_PER_CHAIN
    )

    np.testing.assert_array_equal(precursor.msa_mask, np.ones((3, 2 * N_RES_PER_CHAIN)))
