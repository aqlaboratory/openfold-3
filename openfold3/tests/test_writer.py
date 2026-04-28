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

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from biotite import structure
from biotite.structure.io import pdb, pdbx

from openfold3.core.runners.writer import OF3OutputWriter


@pytest.fixture(params=[np.float16, np.float32])
def dummy_confidence_scores(request):
    dtype = request.param
    n_tokens = 3
    n_atoms = 5

    def rand(*shape):
        return np.random.uniform(size=shape).astype(dtype)

    return {
        "plddt": rand(n_atoms),
        "pde_probs": rand(n_tokens, n_tokens, 64),
        "pde": rand(n_tokens, n_tokens),
        "gpde": rand(1),
        "pae_probs": rand(n_tokens, n_tokens, 64),
        "pae": rand(n_tokens, n_tokens),
        "iptm": rand(1),
        "ptm": rand(1),
        "disorder": rand(1),
        "has_clash": np.dtype(dtype).type(0.0),
        "sample_ranking_score": rand(1),
        "chain_ptm": {
            "1": rand(1),
            "2": rand(1),
        },
        "chain_pair_iptm": {
            "(1, 2)": rand(1),
        },
        "bespoke_iptm": {
            "(1, 2)": rand(1),
        },
    }


class TestPredictionWriter:
    @pytest.mark.parametrize(
        "structure_format",
        ["pdb", "cif", "cif.gz"],
    )
    @pytest.mark.parametrize(
        "file_prefix",
        [
            pytest.param("file_name", id="simple"),
            pytest.param("file.name.with.dots", id="dotted_query_id"),
            pytest.param(
                "file.name.with.cif.and.dots", id="dotted_query_id_with_cif_extension"
            ),
        ],
    )
    def test_written_coordinates(self, tmp_path, structure_format, file_prefix):
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
            structure_format=structure_format,
            full_confidence_output_format="json",
        )
        tmp_file = tmp_path / f"{file_prefix}.{structure_format}"
        output_writer.write_structure_prediction(
            atom_array, new_coords, dummy_plddt, tmp_file, False
        )

        match structure_format:
            case "cif":
                read_file = pdbx.CIFFile.read(tmp_file)
                parsed_structure = pdbx.get_structure(read_file)

            case "cif.gz":
                with gzip.open(tmp_file, "rt") as f:
                    read_file = pdbx.CIFFile.read(f)
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

    def write_confidence_scores(
        self,
        output_path,
        output_fmt,
        output_dtype,
        write_full_output,
        confidence_scores,
    ):
        atom_array = structure.AtomArray(5)
        atom_array.coord = np.zeros((5, 3))
        atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])

        output_writer = OF3OutputWriter(
            output_dir=output_path,
            full_confidence_output_format=output_fmt,
            full_confidence_output_dtype=output_dtype,
            write_full_confidence_scores=write_full_output,
        )
        output_prefix = output_path / "test"
        output_writer.write_confidence_scores(
            confidence_scores, atom_array, output_prefix
        )

    @pytest.mark.parametrize("output_fmt", ["json", "npz"])
    def test_full_confidence_scores_written(
        self, tmp_path, output_fmt, dummy_confidence_scores
    ):
        # infer the dtype from the dummy_confidence_scores fixture
        output_dtype = dummy_confidence_scores["plddt"].dtype.name
        self.write_confidence_scores(
            tmp_path, output_fmt, output_dtype, True, dummy_confidence_scores
        )

        output_prefix = tmp_path / "test"
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
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
            "plddt": dummy_confidence_scores["plddt"],
            "pde": dummy_confidence_scores["pde"],
            "pae": dummy_confidence_scores["pae"],
        }
        actual_full_scores = self._load_full_confidence_scores(out_file_full)

        expected_decimal = 3 if output_dtype == "float16" else 6
        for k in expected_full_scores:
            assert k in actual_full_scores, f"Key {k} not found in actual scores"
            np.testing.assert_array_almost_equal(
                expected_full_scores[k], actual_full_scores[k], decimal=expected_decimal
            )
            if output_fmt == "npz":
                assert actual_full_scores[k].dtype == np.dtype(output_dtype), (
                    f"Expected dtype {output_dtype} for {k}, but got {actual_full_scores[k].dtype}"
                )

    def test_skip_full_confidence_scores(self, tmp_path, dummy_confidence_scores):
        self.write_confidence_scores(
            tmp_path, "json", "float32", False, dummy_confidence_scores
        )
        expected_output_contents = [tmp_path / "test_confidences_aggregated.json"]
        actual_output_contents = [f for f in tmp_path.glob("*")]
        assert expected_output_contents == actual_output_contents, (
            "Only aggregated confidence scores file should be written"
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
