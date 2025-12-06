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

import json
from pathlib import Path

import numpy as np
import pytest  # noqa: F401  - used for pytest tmp fixture
from biotite import structure
from biotite.structure.io import pdb, pdbx

from openfold3.core.runners.writer import OF3OutputWriter

def _create_mock_atom_array(residue_ids: list[str], chain_id="A") -> AtomArray:
    seq_len = len(residue_ids)
    n_atoms = seq_len * 4
    atom_array = structure.AtomArray(n_atoms)

    for i in range(seq_len):
        for j in range(4):
            atom_index = i * 4 + j
            atom_array.res_id[atom_index] = i + 1
            atom_array.ins_code[atom_index] = ''
            atom_array.chain_id[atom_index] = chain_id
            atom_array.atom_name[atom_index] = ['N', 'CA', 'C', 'O'][j]
            atom_array.res_name[atom_index] = 'ALA'

    return atom_array


class TestPredictionWriter:
    @pytest.mark.parametrize(
        "structure_format",
        ["pdb", "cif"],
        ids=lambda x: x,
    )
    def test_written_coordinates(self, tmp_path, structure_format):
        atom1 = structure.Atom([1, 2, 3], chain_id="A")
        atom2 = structure.Atom([2, 3, 4], chain_id="A")
        atom3 = structure.Atom([3, 4, 5], chain_id="B")

        atom_array = structure.array([atom1, atom2, atom3])
        atom_array.entity_id = np.array(["A", "A", "B"])
        atom_array.molecule_type_id = np.array(["0", "0", "1"])
        atom_array.pdbx_formal_charge = np.array(["1", "1", "1"])

        # add extra dimension for sample
        new_coords = np.array(
            [
                [2.0, 2.0, 2.0],
                [3.5, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        dummy_plddt = np.array([0.9, 0.8, 0.7])

        output_writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=False,
            structure_format=structure_format,
            full_confidence_output_format="json",
        )
        tmp_file = tmp_path / f"TEST.{structure_format}"
        output_writer.write_structure_prediction(
            atom_array, new_coords, dummy_plddt, tmp_file, False
        )

        match structure_format:
            case "cif":
                read_file = pdbx.CIFFile.read(tmp_file)
                parsed_structure = pdbx.get_structure(read_file)

            case "pdb":
                parsed_structure = pdb.PDBFile.read(tmp_file).get_structure()

        parsed_coords = parsed_structure.coord[0]
        np.testing.assert_array_equal(parsed_coords, new_coords, strict=False)

    def _load_full_confidence_scores(self, output_file_path):
        output_fmt = output_file_path.suffix.lstrip(".")
        match output_fmt:
            case "json":
                actual_full_scores = json.loads(output_file_path.read_text())
                actual_full_scores = {
                    k: np.array(v) for k, v in actual_full_scores.items()
                }
            case "npz":
                actual_full_scores = np.load(output_file_path)
        return actual_full_scores

    @pytest.mark.parametrize(
        "output_fmt",
        ["json", "npz"],
        ids=lambda x: x,
    )
    def test_confidence_writer_without_pae(
        self, tmp_path, output_fmt, dummy_atom_array
    ):
        n_tokens = 3
        n_atoms = 5
        dummy_atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])

        confidence_scores = {
            "plddt": np.random.uniform(size=n_atoms),
            "pde_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pde": np.random.uniform(size=(n_tokens, n_tokens)),
            "gpde": np.float32(16.2),
        }

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=False,
            full_confidence_output_format=output_fmt,
        )
        output_prefix = tmp_path / "test"
        writer.write_confidence_scores(
            confidence_scores, dummy_atom_array, output_prefix
        )

        # Check aggregated confidence scores
        expected_agg_scores = {
            "avg_plddt": np.mean(confidence_scores["plddt"]),
            "gpde": confidence_scores["gpde"],
        }
        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        actual_agg_scores = json.loads(out_file_agg.read_text())
        assert expected_agg_scores == actual_agg_scores

        # Check full confidence scores:
        expected_full_scores = {
            "plddt": confidence_scores["plddt"],
            "pde": confidence_scores["pde"],
        }
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
        actual_full_scores = self._load_full_confidence_scores(out_file_full)

        for k in expected_full_scores:
            assert k in actual_full_scores, f"Key {k} not found in actual scores"
            np.testing.assert_array_equal(
                expected_full_scores[k], actual_full_scores[k]
            )

    @pytest.mark.parametrize(
        "output_fmt",
        ["json", "npz"],
        ids=lambda x: x,
    )
    def test_confidence_writer_with_pae(self, tmp_path, output_fmt, dummy_atom_array):
        n_tokens = 3
        n_atoms = 5
        dummy_atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])

        confidence_scores = {
            "plddt": np.random.uniform(size=n_atoms),
            "pde_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pde": np.random.uniform(size=(n_tokens, n_tokens)),
            "gpde": np.random.uniform(size=(1,)),
            "pae_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pae": np.random.uniform(size=(n_tokens, n_tokens)),
            "iptm": np.random.uniform(size=(1,)),
            "ptm": np.random.uniform(size=(1,)),
            "disorder": np.random.uniform(size=(1,)),
            "has_clash": np.float32(0.0),
            "sample_ranking_score": np.random.uniform(size=(1,)),
            "chain_ptm": {
                "1": np.random.uniform(size=(1,)),
                "2": np.random.uniform(size=(1,)),
            },
            "chain_pair_iptm": {
                "(1, 2)": np.random.uniform(size=(1,)),
            },
            "bespoke_iptm": {
                "(1, 2)": np.random.uniform(size=(1,)),
            },
        }

        output_writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=True,
            full_confidence_output_format=output_fmt,
        )

        output_prefix = tmp_path / "test"
        output_writer.write_confidence_scores(
            confidence_scores, dummy_atom_array, output_prefix
        )

        expected_agg_score_keys = [
            "avg_plddt",
            "gpde",
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
            "chain_ptm",
            "chain_pair_iptm",
            "bespoke_iptm",
        ]

        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        actual_agg_scores = json.loads(out_file_agg.read_text())
        assert set(expected_agg_score_keys) == set(actual_agg_scores.keys())

        # Check full confidence scores:
        expected_full_scores = {
            "plddt": confidence_scores["plddt"],
            "pde": confidence_scores["pde"],
        }
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
        actual_full_scores = self._load_full_confidence_scores(out_file_full)

        for k in expected_full_scores:
            assert k in actual_full_scores, f"Key {k} not found in actual scores"
            np.testing.assert_array_equal(
                expected_full_scores[k], actual_full_scores[k]
            )

    def test_skips_none_output(self, tmp_path):
        class DummyMock:
            pass

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            structure_format="pdb",
            full_confidence_output_format="npz",
        )
        trainer = DummyMock()
        pl_module = DummyMock()

        writer.on_predict_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch={"query_id": "query_id"},
            batch_idx=0,
        )

        assert writer.failed_count == 1
        assert writer.success_count == 0

    def test_renumber_atom_array_with_insertion_codes(self):
        """Tests OF3OutputWriter._renumber_atom_array to ensure it correctly 
        parses and applies custom residue IDs, including insertion codes (e.g., '101A').
        """

        n_residues = 5
        atoms_per_residue = 2
        n_atoms = n_residues * atoms_per_residue

        atom_array_template = structure.AtomArray(n_atoms)
        atom_array_template.res_id = np.repeat(np.arange(1, n_residues + 1), atoms_per_residue)
        atom_array_template.chain_id = np.repeat(['C'], n_atoms)
        
        atom_array = atom_array_template.copy()

        custom_ids = ['100', '101A', '102', '103B', '104']
        
        renumbered_array = OF3OutputWriter._renumber_atom_array(
            atom_array=atom_array,
            custom_residue_ids=custom_ids
        )
        
        expected_res_id = np.repeat([100, 101, 102, 103, 104], atoms_per_residue)
        np.testing.assert_array_equal(renumbered_array.res_id, expected_res_id)
        
        expected_ins_code = np.array(['', '', 'A', 'A', '', '', 'B', 'B', '', ''], dtype=object)
        assert np.array_equal(renumbered_array.ins_code, expected_ins_code)
        
        atom_array_short = atom_array_template.copy()
        custom_ids_short = ['10', '11'] 

        renumbered_array_short = OF3OutputWriter._renumber_atom_array(
            atom_array=atom_array_short,
            custom_residue_ids=custom_ids_short
        )
        
        expected_res_id_short = np.repeat([10, 11, 0, 0, 0], atoms_per_residue)
        np.testing.assert_array_equal(renumbered_array_short.res_id, expected_res_id_short)


    def test_end_to_end_custom_ids_write(self, tmp_path):
            """Ověřuje celý tok zápisu custom ID (včetně inzerčních kódů) do mmCIF souboru."""
            
            custom_residue_ids = ["1", "2A", "3", "4"] 

            preprocessed_atom_array = _create_mock_atom_array(custom_residue_ids)
            
            mock_batch = {
                "query_id": "test_query_custom_ids",
                "atom_array": preprocessed_atom_array,
                "custom_residue_ids": [custom_residue_ids],
                "plddt": np.repeat(np.array([0.9, 0.8, 0.7, 0.6]), 4)
            }

            
            output_writer = OF3OutputWriter(
                output_dir=tmp_path,
                pae_enabled=False,
                structure_format="cif", 
                full_confidence_output_format="json",
            )
            
            output_writer.on_predict_batch_end(
                trainer=None,
                pl_module=None,
                outputs={}, 
                batch=mock_batch,
                batch_idx=0,
            )

            written_path = tmp_path / "test_query_custom_ids" / "test_query_custom_ids.cif"
            assert written_path.exists()
            
            written_atom_array = pdbx.CIFFile.read(written_path).get_structure()
            
            expected_res_id = np.array([1, 2, 3, 4]) 
            expected_ins_code = np.array(['', 'A', '', ''], dtype=object)

            expanded_expected_res_id = np.repeat(expected_res_id, 4) 
            expanded_expected_ins_code = np.repeat(expected_ins_code, 4)

            np.testing.assert_array_equal(written_atom_array.res_id, expanded_expected_res_id)
            assert np.array_equal(written_atom_array.ins_code, expanded_expected_ins_code)