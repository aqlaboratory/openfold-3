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
from types import SimpleNamespace

import pytest

from openfold3.core.data.primitives.structure.query import CovalentQueryError
from openfold3.core.utils.callbacks import LogInferenceQuerySet
from openfold3.entry_points import experiment_runner as experiment_runner_module
from openfold3.projects.of3_all_atom.config import (
    inference_query_processing as query_processing_module,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Atom,
    InferenceQuerySet,
)
from openfold3.projects.of3_all_atom.config.inference_query_processing import (
    preflight_covalent_query_set,
)


def _query_set(*query_names: str) -> InferenceQuerySet:
    return InferenceQuerySet.model_validate(
        {
            "queries": {
                query_name: {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "C",
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "L",
                            "smiles": "CCl",
                        },
                    ],
                    "covalent_bonds": [[["A", 1, "SG"], ["L", 1, "C1"]]],
                }
                for query_name in query_names
            }
        }
    )


def _prepare(
    query_set: InferenceQuerySet,
    *,
    output_format: str = "cif",
    infer_leaving_atoms: bool = False,
    ccd_file_path=None,
) -> InferenceQuerySet:
    return preflight_covalent_query_set(
        query_set,
        structure_format=output_format,
        infer_leaving_atoms=infer_leaving_atoms,
        ccd_file_path=ccd_file_path,
    )


def test_covalent_query_rejects_pdb_output():
    with pytest.raises(ValueError, match="Select CIF or CIF.GZ"):
        _prepare(_query_set("bonded"), output_format="pdb")


def test_leaving_atom_inference_disabled_preserves_query_set_identity(monkeypatch):
    query_set = _query_set("bonded")
    preflighted = []
    monkeypatch.setattr(
        query_processing_module,
        "structure_with_ref_mols_from_query",
        lambda query: preflighted.append(query.query_name),
    )

    assert _prepare(query_set) is query_set
    assert preflighted == ["bonded"]


def test_leaving_atom_inference_isolates_invalid_queries(monkeypatch):
    query_set = _query_set("valid", "ambiguous")

    def fake_infer(query, ccd=None):
        assert ccd is None
        if query.query_name == "ambiguous":
            raise CovalentQueryError("multiple endpoint-local groups")
        effective_query = query.model_copy(deep=True)
        effective_query.leaving_atoms = [Atom("L", 1, "CL1")]
        return effective_query

    monkeypatch.setattr(query_processing_module, "infer_ccd_leaving_atoms", fake_infer)
    monkeypatch.setattr(
        query_processing_module,
        "structure_with_ref_mols_from_query",
        lambda query: None,
    )

    effective_set = _prepare(query_set, infer_leaving_atoms=True)

    assert list(effective_set.queries) == ["valid"]
    assert effective_set.queries["valid"].leaving_atoms == [Atom("L", 1, "CL1")]
    assert query_set.queries["valid"].leaving_atoms is None


def test_effective_query_log_contains_materialized_leaving_atoms(tmp_path):
    query_set = _query_set("bonded")
    query_set.queries["bonded"].leaving_atoms = [Atom("L", 1, "CL1")]
    pl_module = SimpleNamespace(
        trainer=SimpleNamespace(
            datamodule=SimpleNamespace(
                inference_config=SimpleNamespace(query_set=query_set)
            )
        )
    )

    LogInferenceQuerySet(tmp_path).on_predict_start(None, pl_module)

    logged = json.loads((tmp_path / "inference_query_set.json").read_text())
    assert logged["queries"]["bonded"]["leaving_atoms"] == [["L", 1, "CL1"]]
    assert logged["queries"]["bonded"]["covalent_bonds"] == [
        [["A", 1, "SG"], ["L", 1, "C1"]]
    ]


def test_preflight_passes_custom_text_ccd_only_to_leaving_inference(
    monkeypatch, tmp_path
):
    ccd_path = tmp_path / "components.cif"
    calls = []
    parsed_ccd = object()

    monkeypatch.setattr(
        query_processing_module.pdbx.CIFFile,
        "read",
        lambda path: calls.append(("parse", path)) or parsed_ccd,
    )

    monkeypatch.setattr(
        query_processing_module,
        "infer_ccd_leaving_atoms",
        lambda query, ccd=None: calls.append(("infer", query.query_name, ccd)) or query,
    )
    monkeypatch.setattr(
        query_processing_module,
        "structure_with_ref_mols_from_query",
        lambda query: calls.append(("build", query.query_name)),
    )

    _prepare(
        _query_set("bonded_1", "bonded_2"),
        infer_leaving_atoms=True,
        ccd_file_path=ccd_path,
    )

    assert calls == [
        ("parse", ccd_path),
        ("infer", "bonded_1", parsed_ccd),
        ("build", "bonded_1"),
        ("infer", "bonded_2", parsed_ccd),
        ("build", "bonded_2"),
    ]


def test_leaving_atom_inference_rejects_job_when_all_queries_fail(monkeypatch):
    def fake_infer(query, ccd=None):
        raise CovalentQueryError("multiple endpoint-local groups")

    monkeypatch.setattr(query_processing_module, "infer_ccd_leaving_atoms", fake_infer)

    with pytest.raises(ValueError, match="No valid queries remain"):
        _prepare(_query_set("ambiguous"), infer_leaving_atoms=True)


def test_structural_preflight_isolates_invalid_queries(monkeypatch):
    query_set = _query_set("valid", "invalid")

    def fake_build(query):
        if query.query_name == "invalid":
            raise CovalentQueryError("unknown atom name")

    monkeypatch.setattr(
        query_processing_module, "structure_with_ref_mols_from_query", fake_build
    )

    effective_set = _prepare(query_set)

    assert list(effective_set.queries) == ["valid"]


def test_structural_preflight_does_not_hide_unexpected_failures(monkeypatch):
    monkeypatch.setattr(
        query_processing_module,
        "structure_with_ref_mols_from_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("implementation bug")),
    )

    with pytest.raises(RuntimeError, match="implementation bug"):
        _prepare(_query_set("bonded"))


def test_manual_leaving_atoms_are_normalized_without_automatic_inference(monkeypatch):
    query_set = _query_set("bonded")
    query_set.queries["bonded"].leaving_atoms = [
        Atom("L", 1, "CL1"),
        Atom("L", 1, "CL1"),
    ]
    monkeypatch.setattr(
        query_processing_module,
        "structure_with_ref_mols_from_query",
        lambda query: None,
    )

    effective_set = _prepare(query_set)

    assert effective_set.queries["bonded"].leaving_atoms == [Atom("L", 1, "CL1")]
    assert len(query_set.queries["bonded"].leaving_atoms) == 2


def test_load_inference_runner_args_returns_query_preflight_settings(tmp_path):
    ccd_path = tmp_path / "components.cif"
    ccd_path.touch()
    runner_yaml = tmp_path / "runner.yaml"
    runner_yaml.write_text(
        "output_writer_settings:\n"
        "  structure_format: cif.gz\n"
        "dataset_config_kwargs:\n"
        f"  ccd_file_path: {ccd_path}\n"
    )
    runner_args, settings, default_path = (
        experiment_runner_module.load_inference_runner_args(runner_yaml)
    )

    assert runner_args["output_writer_settings"]["structure_format"] == "cif.gz"
    assert settings.structure_format == "cif.gz"
    assert settings.ccd_file_path == ccd_path
    assert default_path is None
