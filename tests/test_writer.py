import json
from pathlib import Path

import numpy as np
import pytest  # noqa: F401  - used for pytest tmp fixture
from biotite import structure
from biotite.structure.io import pdb, pdbx

from openfold3.core.runners.writer import OF3OutputWriter


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
            structure_format=structure_format,
            full_confidence_output_format="json",
        )
        tmp_file = tmp_path / f"TEST.{structure_format}"
        output_writer.write_structure_prediction(
            atom_array, new_coords, dummy_plddt, tmp_file
        )

        match structure_format:
            case "cif":
                read_file = pdbx.CIFFile.read(tmp_file)
                parsed_structure = pdbx.get_structure(read_file)

            case "pdb":
                parsed_structure = pdb.PDBFile.read(tmp_file).get_structure()

        parsed_coords = parsed_structure.coord[0]
        np.testing.assert_array_equal(parsed_coords, new_coords, strict=False)

    @pytest.mark.parametrize(
        "output_fmt",
        ["json", "npz"],
        ids=lambda x: x,
    )
    def test_written_confidence_scores(self, tmp_path, output_fmt):
        n_tokens = 3
        n_atoms = 5
        confidence_scores = {
            "plddt": np.random.uniform(size=n_atoms),
            "distance_confidence_probs": np.random.uniform(
                size=(n_tokens, n_tokens, 64)
            ),
            "predicted_distance_error": np.random.uniform(size=(n_tokens, n_tokens)),
            "max_predicted_distance_error": np.float32(15.2),
            "global_predicted_distance_error": np.float32(16.2),
        }

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            structure_format="pdb",
            full_confidence_output_format=output_fmt,
        )
        output_prefix = tmp_path / "test_"
        writer.write_confidence_scores(confidence_scores, output_prefix)

        # Check aggregated confidence scores
        expected_agg_scores = {
            "avg_plddt": np.mean(confidence_scores["plddt"]),
            "gpde": confidence_scores["global_predicted_distance_error"],
        }
        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        actual_agg_scores = json.loads(out_file_agg.read_text())
        assert expected_agg_scores == actual_agg_scores

        # Check full confidence scores:
        expected_full_scores = {
            "plddt": confidence_scores["plddt"],
            "pde": confidence_scores["predicted_distance_error"],
        }
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
        match output_fmt:
            case "json":
                actual_full_scores = json.loads(out_file_full.read_text())
                actual_full_scores = {
                    k: np.array(v) for k, v in actual_full_scores.items()
                }
            case "npz":
                actual_full_scores = np.load(out_file_full)

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
            self,
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch={},
            batch_idx=0,
        )

        assert writer.failed_count == 1
        assert writer.success_count == 0
