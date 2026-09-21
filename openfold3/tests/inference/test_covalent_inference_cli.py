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

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from click.testing import CliRunner

from openfold3.run_openfold import cli

COVALENT_EXAMPLE_DIRECTORY = (
    Path(__file__).parents[3] / "examples" / "example_inference_inputs"
)


@pytest.mark.parametrize(
    "example_name",
    [
        "query_covalent_ccd_0e6.json",
        "query_covalent_nag_serine.json",
        "query_covalent_smiles.json",
        "query_covalent_polymer_polymer.json",
    ],
)
def test_covalent_documentation_examples_match_query_schema(example_name):
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    example_path = COVALENT_EXAMPLE_DIRECTORY / example_name
    InferenceQuerySet.from_json(example_path)


@pytest.mark.parametrize(
    ("example_name", "infer_leaving_atoms"),
    [
        ("query_covalent_ccd_0e6.json", True),
        ("query_covalent_nag_serine.json", True),
        ("query_covalent_smiles.json", False),
        ("query_covalent_polymer_polymer.json", False),
    ],
)
def test_covalent_documentation_examples_resolve_declared_bonds(
    example_name, infer_leaving_atoms
):
    from openfold3.core.data.primitives.structure.query import (
        infer_ccd_leaving_atoms,
        structure_with_ref_mols_from_query,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    query_set = InferenceQuerySet.from_json(COVALENT_EXAMPLE_DIRECTORY / example_name)
    for query in query_set.queries.values():
        if infer_leaving_atoms:
            query = infer_ccd_leaving_atoms(query)
        atom_array = structure_with_ref_mols_from_query(query).atom_array
        bond_pairs = {
            frozenset((int(atom_1), int(atom_2)))
            for atom_1, atom_2, _bond_type in atom_array.bonds.as_array()
        }
        for endpoint_1, endpoint_2 in query.covalent_bonds:
            endpoint_indices = []
            for endpoint in (endpoint_1, endpoint_2):
                matches = [
                    index
                    for index, (chain_id, residue_id, atom_name) in enumerate(
                        zip(
                            atom_array.chain_id,
                            atom_array.res_id,
                            atom_array.atom_name,
                            strict=True,
                        )
                    )
                    if (chain_id, residue_id, atom_name) == endpoint
                ]
                assert len(matches) == 1
                endpoint_indices.append(matches[0])
            assert frozenset(endpoint_indices) in bond_pairs


def test_covalent_runner_example_matches_configuration_schema():
    from openfold3.core.config.config_utils import load_yaml
    from openfold3.entry_points.validator import (
        InferenceExperimentSettings,
        OutputWritingSettings,
    )

    example_path = (
        Path(__file__).parents[3]
        / "examples"
        / "example_runner_yamls"
        / "covalent_bonds.yml"
    )
    config = load_yaml(example_path)

    InferenceExperimentSettings.model_validate(config.get("experiment_settings", {}))
    output_settings = OutputWritingSettings.model_validate(
        config["output_writer_settings"]
    )
    assert (
        "infer_covalent_leaving_atoms" not in InferenceExperimentSettings.model_fields
    )
    assert output_settings.structure_format == "cif"


def test_leaving_atom_inference_cannot_be_enabled_in_runner_yaml():
    from openfold3.entry_points.validator import InferenceExperimentSettings

    with pytest.raises(ValueError, match="CLI-only"):
        InferenceExperimentSettings.model_validate(
            {"infer_covalent_leaving_atoms": True}
        )


def test_inspect_molecule_reports_production_atom_names_and_graph():
    result = CliRunner().invoke(
        cli,
        ["inspect-molecule", "--smiles", "CC(=O)Cl"],
    )

    assert result.exit_code == 0, result.output
    atoms = json.loads(result.output)
    assert [atom["atom_name"] for atom in atoms] == ["C1", "C2", "O1", "CL1"]
    assert [atom["diagnostic_atom_index_not_selector"] for atom in atoms] == [
        0,
        1,
        2,
        3,
    ]
    assert atoms[1] == {
        "atom_name": "C2",
        "element": "C",
        "formal_charge": 0,
        "aromatic": False,
        "neighbors": [
            {"atom_name": "C1", "bond_type": "SINGLE"},
            {"atom_name": "CL1", "bond_type": "SINGLE"},
            {"atom_name": "O1", "bond_type": "DOUBLE"},
        ],
        "diagnostic_atom_index_not_selector": 1,
    }


def test_inspect_molecule_help_describes_json_output_and_selector_semantics():
    result = CliRunner().invoke(cli, ["inspect-molecule", "--help"])

    assert result.exit_code == 0, result.output
    assert "JSON array" in result.output
    assert "generated atom names" in result.output
    assert "must not be used as a query selector" in result.output


def test_inspect_molecule_writes_json_file(tmp_path):
    output_path = tmp_path / "molecule.json"
    result = CliRunner().invoke(
        cli,
        [
            "inspect-molecule",
            "--smiles",
            "C",
            "--output-json",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert json.loads(output_path.read_text()) == [
        {
            "atom_name": "C1",
            "element": "C",
            "formal_charge": 0,
            "aromatic": False,
            "neighbors": [],
            "diagnostic_atom_index_not_selector": 0,
        }
    ]


def test_inspect_molecule_reports_invalid_smiles():
    result = CliRunner().invoke(
        cli,
        ["inspect-molecule", "--smiles", "not a valid smiles"],
    )

    assert result.exit_code != 0
    assert "Could not construct a molecule" in result.output


@pytest.mark.parametrize(
    "flag, expected",
    [
        ([], False),
        (["--infer-covalent-leaving-atoms"], True),
    ],
)
def test_predict_forwards_infer_covalent_leaving_atoms(
    monkeypatch, tmp_path, flag, expected
):
    captured = {}

    class FakeInferenceExperimentConfig:
        def __init__(self, **kwargs):
            assert captured["prepared_before_config"] is True

    class FakeInferenceExperimentRunner:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def setup(self):
            assert captured["prepared_before_config"] is True

        def run(self, query_set):
            captured["ran_query_set"] = query_set

        def cleanup(self):
            pass

    class FakeInferenceQuerySet:
        queries = {"query": object()}

    def fake_load_query_set(json_path):
        assert captured["runner_args_prepared"] is True
        return FakeInferenceQuerySet(), {}

    def fake_preflight_covalent_query_set(
        query_set, *, structure_format, ccd_file_path, infer_leaving_atoms
    ):
        captured["prepared_before_config"] = True
        captured["preflight_flag"] = infer_leaving_atoms
        captured["preflight_structure_format"] = structure_format
        captured["preflight_ccd_file_path"] = ccd_file_path
        captured["prepared_query_set"] = query_set
        return query_set

    runner_module = ModuleType("openfold3.entry_points.experiment_runner")
    runner_module.InferenceExperimentRunner = FakeInferenceExperimentRunner

    def fake_load_runner_args(runner_yaml, default_yaml):
        captured["runner_args_prepared"] = True
        return (
            {},
            SimpleNamespace(structure_format="cif", ccd_file_path=None),
            None,
        )

    runner_module.load_inference_runner_args = fake_load_runner_args
    validator_module = ModuleType("openfold3.entry_points.validator")
    validator_module.InferenceExperimentConfig = FakeInferenceExperimentConfig
    query_processing_module = ModuleType(
        "openfold3.projects.of3_all_atom.config.inference_query_processing"
    )
    query_processing_module.load_inference_query_set_with_query_errors = (
        fake_load_query_set
    )
    query_processing_module.preflight_covalent_query_set = (
        fake_preflight_covalent_query_set
    )
    monkeypatch.setitem(
        sys.modules, "openfold3.entry_points.experiment_runner", runner_module
    )
    monkeypatch.setitem(
        sys.modules, "openfold3.entry_points.validator", validator_module
    )
    monkeypatch.setitem(
        sys.modules,
        "openfold3.projects.of3_all_atom.config.inference_query_processing",
        query_processing_module,
    )

    query_json = tmp_path / "query.json"
    query_json.write_text("{}")
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--query-json",
            str(query_json),
            "--use_tf32",
            "false",
            *flag,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["preflight_flag"] is expected
    assert captured["preflight_structure_format"] == "cif"
    assert captured["preflight_ccd_file_path"] is None
    assert captured["prepared_before_config"] is True
    assert captured["ran_query_set"] is captured["prepared_query_set"]
    assert "infer_covalent_leaving_atoms" not in captured


def test_predict_has_no_negative_leaving_atom_inference_flag(tmp_path):
    query_json = tmp_path / "query.json"
    query_json.write_text("{}")

    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--query-json",
            str(query_json),
            "--no-infer-covalent-leaving-atoms",
        ],
    )

    assert result.exit_code != 0
    assert "No such option: --no-infer-covalent-leaving-atoms" in result.output
