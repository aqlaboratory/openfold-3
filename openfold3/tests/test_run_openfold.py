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

import pytest
from click.testing import CliRunner

import openfold3.run_openfold as run_openfold


@pytest.fixture
def cli_runner():
    return CliRunner()


def _write_query(tmp_path, data):
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps(data))
    return query_path


def test_check_query_text_summary_counts_expanded_chains(tmp_path, cli_runner):
    query_path = _write_query(
        tmp_path,
        {
            "queries": {
                "mixed": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A", "B"],
                            "sequence": "ACD",
                        },
                        {
                            "molecule_type": "rna",
                            "chain_ids": "C",
                            "sequence": "AGCU",
                        },
                        {
                            "molecule_type": "dna",
                            "chain_ids": "D",
                            "sequence": "GACT",
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["E", "F"],
                            "smiles": "CCO",
                        },
                    ]
                }
            }
        },
    )

    result = cli_runner.invoke(
        run_openfold.cli, ["check-query", "--query-json", str(query_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Query file is valid." in result.output
    assert "Queries: 1" in result.output
    assert "Chain declarations: 4" in result.output
    assert "Chain instances: 6" in result.output
    assert "Polymer residues: 14" in result.output
    assert "Ligand chain instances: 2" in result.output


def test_check_query_json_summary_for_multiple_queries(tmp_path, cli_runner):
    query_path = _write_query(
        tmp_path,
        {
            "queries": {
                "protein": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "ACDE",
                        }
                    ]
                },
                "ligand": {
                    "chains": [
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "L",
                            "ccd_codes": "ATP",
                        }
                    ]
                },
            }
        },
    )

    result = cli_runner.invoke(
        run_openfold.cli,
        ["check-query", "--query_json", str(query_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["valid"] is True
    assert output["summary"]["query_count"] == 2
    assert output["summary"]["molecule_type_counts"] == {
        "ligand": 1,
        "protein": 1,
    }
    assert output["summary"]["ligand_representation_counts"] == {"ccd_codes": 1}


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        ("{not json", "Query validation failed"),
        (
            json.dumps(
                {
                    "queries": {
                        "bad": {
                            "chains": [
                                {
                                    "molecule_type": "protein",
                                    "chain_ids": "A",
                                    "sequence": "ACD",
                                    "unknown_field": True,
                                }
                            ]
                        }
                    }
                }
            ),
            "unknown_field",
        ),
        (
            json.dumps(
                {
                    "queries": {
                        "bad": {
                            "chains": [
                                {
                                    "molecule_type": "carbohydrate",
                                    "chain_ids": "A",
                                    "sequence": "ACD",
                                }
                            ]
                        }
                    }
                }
            ),
            "molecule_type",
        ),
    ],
)
def test_check_query_reports_validation_errors(
    tmp_path, cli_runner, capsys, contents, expected_error
):
    query_path = tmp_path / "query.json"
    query_path.write_text(contents)

    result = cli_runner.invoke(
        run_openfold.cli, ["check-query", "--query-json", str(query_path)]
    )

    assert result.exit_code != 0
    captured = capsys.readouterr()
    assert expected_error in (
        result.output + result.stderr + captured.out + captured.err
    )


def test_check_query_rejects_missing_referenced_path(tmp_path, cli_runner):
    query_path = _write_query(
        tmp_path,
        {
            "queries": {
                "protein": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "ACD",
                            "main_msa_file_paths": str(tmp_path / "missing.a3m"),
                        }
                    ]
                }
            }
        },
    )

    result = cli_runner.invoke(
        run_openfold.cli, ["check-query", "--query-json", str(query_path)]
    )

    assert result.exit_code != 0
    assert "main_msa_file_paths" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["check-query"],
        ["check-query", "--query-json", "query.json", "--format", "xml"],
    ],
)
def test_check_query_rejects_invalid_cli_arguments(cli_runner, args):
    result = cli_runner.invoke(run_openfold.cli, args)

    assert result.exit_code != 0


def test_check_query_does_not_initialize_gpu(tmp_path, cli_runner, monkeypatch):
    query_path = _write_query(
        tmp_path,
        {
            "queries": {
                "protein": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "ACD",
                        }
                    ]
                }
            }
        },
    )

    def fail_gpu_setup():
        raise AssertionError("GPU setup must not be called")

    monkeypatch.setattr(run_openfold, "_torch_gpu_setup", fail_gpu_setup)
    result = cli_runner.invoke(
        run_openfold.cli, ["check-query", "--query-json", str(query_path)]
    )

    assert result.exit_code == 0, result.output
