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

"""Tests for custom residue numbering (issue #58).

These tests verify that:
1. The Chain Pydantic model accepts the new starting_residue_number field
2. The writer correctly applies residue number offsets to output files
"""

import numpy as np
import pytest
from biotite import structure
from biotite.structure.io import pdbx
from pydantic import ValidationError

from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    Query,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

# Real mini-protein / peptide sequences used across tests:
#   TRH        = "QHP"          (3 aa, thyrotropin-releasing hormone)
#   OXYTOCIN   = "CYIQNCPLG"   (9 aa, nonapeptide hormone)
#   CHIGNOLIN  = "GYDPETGTWG"  (10 aa, smallest folding protein)
TRH = "QHP"
OXYTOCIN = "CYIQNCPLG"
CHIGNOLIN = "GYDPETGTWG"


class TestChainStartingResidueNumber:
    """Tests for the starting_residue_number field on the Chain Pydantic model."""

    def test_chain_accepts_starting_residue_number(self):
        """Chain model should accept an integer starting_residue_number."""
        chain = Chain.model_validate(
            {
                "molecule_type": "protein",
                "chain_ids": ["A"],
                "sequence": TRH,
                "starting_residue_number": 25,
            }
        )
        assert chain.starting_residue_number == 25

    def test_chain_defaults_to_none(self):
        """Omitting starting_residue_number should default to None (backward compat)."""
        chain = Chain.model_validate(
            {
                "molecule_type": "protein",
                "chain_ids": ["A"],
                "sequence": TRH,
            }
        )
        assert chain.starting_residue_number is None

    def test_chain_rejects_invalid_type(self):
        """Non-integer starting_residue_number should raise ValidationError."""
        with pytest.raises(ValidationError):
            Chain.model_validate(
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": TRH,
                    "starting_residue_number": "abc",
                }
            )

    def test_chain_allows_negative_numbers(self):
        """PDB allows negative residue numbers (e.g., signal peptides)."""
        chain = Chain.model_validate(
            {
                "molecule_type": "protein",
                "chain_ids": ["A"],
                "sequence": TRH,
                "starting_residue_number": -5,
            }
        )
        assert chain.starting_residue_number == -5

    def test_full_query_roundtrip(self):
        """Full Query with starting_residue_number should validate and serialize."""
        query = Query.model_validate(
            {
                "query_name": "test_query",
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "C"],
                        "sequence": CHIGNOLIN,
                        "starting_residue_number": 102,
                    },
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["B"],
                        "sequence": OXYTOCIN,
                        "starting_residue_number": 22,
                    },
                ],
            }
        )
        assert query.chains[0].starting_residue_number == 102
        assert query.chains[1].starting_residue_number == 22

        # Roundtrip through JSON serialization
        json_data = query.model_dump(mode="json")
        query_roundtrip = Query.model_validate(json_data)
        assert query_roundtrip.chains[0].starting_residue_number == 102
        assert query_roundtrip.chains[1].starting_residue_number == 22


# ---------------------------------------------------------------------------
# Helper to create a minimal atom array for writer tests
# ---------------------------------------------------------------------------


def _make_atom_array(chain_ids, res_ids):
    """Create a minimal AtomArray with the given chain_ids and res_ids.

    Args:
        chain_ids: list of chain ID strings, one per atom.
        res_ids: list of residue ID ints, one per atom.

    Returns:
        A biotite AtomArray with the required annotations for write_structure.
    """
    n_atoms = len(chain_ids)
    atoms = []
    for i in range(n_atoms):
        atom = structure.Atom(
            coord=[float(i), 0.0, 0.0],
            chain_id=chain_ids[i],
            res_id=res_ids[i],
            res_name="ALA",
            atom_name="CA",
            element="C",
        )
        atoms.append(atom)

    atom_array = structure.array(atoms)
    atom_array.set_annotation(
        "entity_id", np.array([chain_ids[i] for i in range(n_atoms)])
    )
    atom_array.set_annotation("molecule_type_id", np.array(["0"] * n_atoms))
    atom_array.set_annotation("pdbx_formal_charge", np.array(["0"] * n_atoms))

    return atom_array


def _read_res_ids_from_cif(cif_path):
    """Read back residue IDs from a written CIF file.

    Returns:
        tuple of (chain_ids, res_ids) as numpy arrays.
    """
    read_file = pdbx.CIFFile.read(cif_path)
    parsed = pdbx.get_structure(read_file)
    return parsed.chain_id, parsed.res_id


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWriterResidueNumberOffsets:
    """Tests that write_structure_prediction correctly applies residue offsets."""

    def test_writer_applies_single_chain_offset(self, tmp_path):
        """A single chain with starting_residue_number=25 should produce res_ids
        starting at 25."""
        # 3 residues in chain A: originally res_id 1, 2, 3
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "A"],
            res_ids=[1, 2, 3],
        )
        new_coords = atom_array.coord.copy()
        plddt = np.array([0.9, 0.8, 0.7])

        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=new_coords,
            plddt=plddt,
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={"A": 24},  # starting_residue_number=25 → offset=24
        )

        chain_ids, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids, [25, 26, 27])

    def test_writer_no_offset_unchanged(self, tmp_path):
        """Without offsets, res_id should remain at original values."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "A"],
            res_ids=[1, 2, 3],
        )
        new_coords = atom_array.coord.copy()
        plddt = np.array([0.9, 0.8, 0.7])

        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=new_coords,
            plddt=plddt,
            output_file=output_file,
            make_ost_compatible=False,
        )

        _, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids, [1, 2, 3])

    def test_writer_multimer_different_offsets(self, tmp_path):
        """Two chains with different offsets should both be renumbered correctly."""
        # Chain A: res_ids 1,2,3 with offset 24 → 25,26,27
        # Chain B: res_ids 1,2 with offset 21 → 22,23
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "A", "B", "B"],
            res_ids=[1, 2, 3, 1, 2],
        )
        new_coords = atom_array.coord.copy()
        plddt = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=new_coords,
            plddt=plddt,
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={"A": 24, "B": 21},
        )

        chain_ids, res_ids = _read_res_ids_from_cif(output_file)

        # Chain A atoms should be 25, 26, 27
        chain_a_mask = chain_ids == "A"
        np.testing.assert_array_equal(res_ids[chain_a_mask], [25, 26, 27])

        # Chain B atoms should be 22, 23
        chain_b_mask = chain_ids == "B"
        np.testing.assert_array_equal(res_ids[chain_b_mask], [22, 23])


# ---------------------------------------------------------------------------
# Collator tests — the exact bug that crashed the pipeline
# ---------------------------------------------------------------------------


class TestCollatorResidueNumberOffsets:
    """Tests that openfold_batch_collator handles residue_number_offsets correctly.

    residue_number_offsets is a plain Python dict, not a tensor. If it reaches
    pad_sequence it raises: TypeError: expected Tensor ... but got int.
    """

    def test_collator_survives_with_offsets(self):
        """Collator should not crash when samples contain residue_number_offsets."""
        import torch

        from openfold3.core.data.framework.data_module import (
            openfold_batch_collator,
        )

        sample = {
            "token_mask": torch.ones(5),
            "residue_number_offsets": {"A": 24},
        }
        # Should not raise TypeError
        batch = openfold_batch_collator([sample])
        assert "residue_number_offsets" in batch
        assert batch["residue_number_offsets"] == [{"A": 24}]

    def test_collator_survives_with_empty_offsets(self):
        """Collator should handle empty offsets dict (no starting_residue_number)."""
        import torch

        from openfold3.core.data.framework.data_module import (
            openfold_batch_collator,
        )

        sample = {
            "token_mask": torch.ones(5),
            "residue_number_offsets": {},
        }
        batch = openfold_batch_collator([sample])
        assert batch["residue_number_offsets"] == [{}]

    def test_collator_preserves_per_sample_offsets(self):
        """With batch_size > 1, each sample's offsets should be preserved."""
        import torch

        from openfold3.core.data.framework.data_module import (
            openfold_batch_collator,
        )

        samples = [
            {
                "token_mask": torch.ones(5),
                "residue_number_offsets": {"A": 24, "B": 21},
            },
            {
                "token_mask": torch.ones(5),
                "residue_number_offsets": {"C": 100},
            },
        ]
        batch = openfold_batch_collator(samples)
        assert batch["residue_number_offsets"] == [
            {"A": 24, "B": 21},
            {"C": 100},
        ]

    def test_collator_without_offsets_key(self):
        """Collator should work normally when residue_number_offsets is absent."""
        import torch

        from openfold3.core.data.framework.data_module import (
            openfold_batch_collator,
        )

        sample = {"token_mask": torch.ones(5)}
        batch = openfold_batch_collator([sample])
        assert "residue_number_offsets" not in batch


# ---------------------------------------------------------------------------
# Writer does not mutate the shared atom_array
# ---------------------------------------------------------------------------


class TestWriterOffsetDoesNotMutateOriginal:
    """Calling write_structure_prediction multiple times (as happens in the
    diffusion sample loop) must not accumulate offsets on the shared array."""

    def test_repeated_writes_do_not_accumulate_offset(self, tmp_path):
        """Two consecutive writes with the same offset should produce identical
        numbering, proving the original array was not mutated."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "A"],
            res_ids=[1, 2, 3],
        )
        offsets = {"A": 24}

        for i in range(3):
            output_file = tmp_path / f"sample_{i}.cif"
            OF3OutputWriter.write_structure_prediction(
                atom_array=atom_array,
                predicted_coords=atom_array.coord.copy(),
                plddt=np.array([0.9, 0.8, 0.7]),
                output_file=output_file,
                make_ost_compatible=False,
                residue_number_offsets=offsets,
            )

        # All three files should have identical numbering (25, 26, 27)
        for i in range(3):
            _, res_ids = _read_res_ids_from_cif(tmp_path / f"sample_{i}.cif")
            np.testing.assert_array_equal(
                res_ids,
                [25, 26, 27],
                err_msg=f"Sample {i} has wrong numbering — offset accumulated!",
            )

        # Original atom_array should still have 1, 2, 3
        np.testing.assert_array_equal(atom_array.res_id, [1, 2, 3])


# ---------------------------------------------------------------------------
# Writer edge cases
# ---------------------------------------------------------------------------


class TestWriterEdgeCases:
    """Edge cases for residue number offset application."""

    def test_zero_offset_is_noop(self, tmp_path):
        """starting_residue_number=1 produces offset=0, which should be a no-op."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A"],
            res_ids=[1, 2],
        )
        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=atom_array.coord.copy(),
            plddt=np.array([0.9, 0.8]),
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={"A": 0},
        )
        _, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids, [1, 2])

    def test_negative_offset(self, tmp_path):
        """Negative starting_residue_number (e.g., signal peptides at -5)."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "A"],
            res_ids=[1, 2, 3],
        )
        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=atom_array.coord.copy(),
            plddt=np.array([0.9, 0.8, 0.7]),
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={"A": -6},  # starting_residue_number=-5 → offset=-6
        )
        _, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids, [-5, -4, -3])

    def test_empty_offsets_dict_is_noop(self, tmp_path):
        """An empty offsets dict (no chains specified) should not change numbering."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A"],
            res_ids=[1, 2],
        )
        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=atom_array.coord.copy(),
            plddt=np.array([0.9, 0.8]),
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={},
        )
        _, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids, [1, 2])

    def test_partial_offset_only_affects_specified_chain(self, tmp_path):
        """Offset for chain A only; chain B should stay at original numbering."""
        atom_array = _make_atom_array(
            chain_ids=["A", "A", "B", "B"],
            res_ids=[1, 2, 1, 2],
        )
        output_file = tmp_path / "test.cif"
        OF3OutputWriter.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=atom_array.coord.copy(),
            plddt=np.array([0.9, 0.8, 0.7, 0.6]),
            output_file=output_file,
            make_ost_compatible=False,
            residue_number_offsets={"A": 99},  # Only chain A gets offset
        )
        chain_ids, res_ids = _read_res_ids_from_cif(output_file)
        np.testing.assert_array_equal(res_ids[chain_ids == "A"], [100, 101])
        np.testing.assert_array_equal(res_ids[chain_ids == "B"], [1, 2])


# ---------------------------------------------------------------------------
# Offset computation from Query objects
# ---------------------------------------------------------------------------


class TestOffsetComputation:
    """Tests that the offset dict is computed correctly from Query chains."""

    @staticmethod
    def _compute_offsets(query: Query) -> dict[str, int]:
        """Replicate the offset computation logic from InferenceDataset."""
        residue_number_offsets = {}
        for chain in query.chains:
            if chain.starting_residue_number is not None:
                offset = chain.starting_residue_number - 1
                for chain_id in chain.chain_ids:
                    residue_number_offsets[chain_id] = offset
        return residue_number_offsets

    def test_single_chain_offset(self):
        """Single chain with starting_residue_number=25 → offset 24."""
        query = Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": TRH,
                        "starting_residue_number": 25,
                    }
                ]
            }
        )
        offsets = self._compute_offsets(query)
        assert offsets == {"A": 24}

    def test_homomer_shares_offset(self):
        """A homomer (chain_ids [A, B]) should give both chains the same offset."""
        query = Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "B"],
                        "sequence": TRH,
                        "starting_residue_number": 102,
                    }
                ]
            }
        )
        offsets = self._compute_offsets(query)
        assert offsets == {"A": 101, "B": 101}

    def test_no_offset_produces_empty_dict(self):
        """When no chain specifies starting_residue_number, offsets dict is empty."""
        query = Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": TRH,
                    }
                ]
            }
        )
        offsets = self._compute_offsets(query)
        assert offsets == {}

    def test_mixed_offset_and_no_offset(self):
        """One chain with offset, another without — only the first gets an offset."""
        query = Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": TRH,
                        "starting_residue_number": 50,
                    },
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["B"],
                        "sequence": OXYTOCIN,
                        # no starting_residue_number
                    },
                ]
            }
        )
        offsets = self._compute_offsets(query)
        assert offsets == {"A": 49}
        assert "B" not in offsets
