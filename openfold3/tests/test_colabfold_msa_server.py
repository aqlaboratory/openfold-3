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

import numpy as np
import pytest

from openfold3.core.data.tools import colabfold_msa_server as msa_server
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_riboseek_result_to_a3m_builds_query_centered_rows():
    result = {
        "results": [
            {
                "alignments": {
                    "0": [
                        {
                            "target": "rna_hit description",
                            "qStartPos": 2,
                            "qEndPos": 5,
                            "qAln": "CG-G",
                            "dbAln": "C-AG",
                        }
                    ]
                }
            }
        ]
    }

    a3m = msa_server._riboseek_result_to_a3m("ACGGU", result)

    assert a3m == ">query\nACGGU\n>rna_hit\n-C-AG\n"


def test_riboseek_result_to_a3m_returns_query_only_without_hits():
    assert msa_server._riboseek_result_to_a3m("ACGT", {"results": []}) == (
        ">query\nACGU\n"
    )


def test_query_riboseek_msa_server_polls_and_fetches_results(monkeypatch):
    post_calls = []
    get_urls = []

    def fake_post(url, data, timeout, headers):
        post_calls.append((url, data, timeout, headers))
        return _Response({"id": "ticket-1", "status": "PENDING"})

    def fake_get(url, timeout, headers):
        get_urls.append(url)
        if url.endswith("/api/ticket/ticket-1"):
            return _Response({"id": "ticket-1", "status": "COMPLETE"})
        return _Response(
            {
                "results": [
                    {
                        "alignments": [
                            {
                                "target": "hit",
                                "qStartPos": 1,
                                "qEndPos": 4,
                                "qAln": "GCGG",
                                "dbAln": "GC-G",
                            }
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr(msa_server.requests, "post", fake_post)
    monkeypatch.setattr(msa_server.requests, "get", fake_get)
    monkeypatch.setattr(msa_server.time, "sleep", lambda _: None)

    a3m_lines = msa_server.query_riboseek_msa_server(
        ["GCGG"],
        user_agent="test-agent",
        host_url="https://search.foldseek.com/",
        database="rnadb",
    )

    assert a3m_lines == [">query\nGCGG\n>hit\nGC-G\n"]
    assert post_calls[0][0] == "https://search.foldseek.com/api/ticket/riboseek"
    assert ("database[]", "rnadb") in post_calls[0][1]
    assert get_urls == [
        "https://search.foldseek.com/api/ticket/ticket-1",
        "https://search.foldseek.com/api/result/ticket-1/0",
    ]


def test_query_riboseek_msa_server_times_out_queued_job(monkeypatch):
    monkeypatch.setattr(
        msa_server.requests,
        "post",
        lambda *args, **kwargs: _Response({"id": "ticket-1", "status": "PENDING"}),
    )
    monkeypatch.setattr(
        msa_server.requests,
        "get",
        lambda *args, **kwargs: _Response(
            {"id": "ticket-1", "status": "PENDING"}
        ),
    )
    monotonic_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(msa_server.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(msa_server.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="timed out"):
        msa_server.query_riboseek_msa_server(
            ["GCGG"],
            user_agent="test-agent",
            job_timeout_seconds=1,
        )


def test_preprocess_colabfold_msas_adds_protein_and_rna_paths(tmp_path, monkeypatch):
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "job": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "AAAA",
                        },
                        {
                            "molecule_type": "rna",
                            "chain_ids": ["B"],
                            "sequence": "GCGG",
                        },
                    ],
                }
            }
        }
    )

    monkeypatch.setattr(
        msa_server,
        "query_colabfold_msa_server",
        lambda *args, **kwargs: [">101\nAAAA\n>protein_hit\nAA-A\n"],
    )
    monkeypatch.setattr(
        msa_server,
        "query_riboseek_msa_server",
        lambda *args, **kwargs: [">query\nGCGG\n>rna_hit\nGC-G\n"],
    )

    settings = msa_server.MsaComputationSettings(
        msa_output_directory=tmp_path,
        msa_file_format="npz",
        save_mappings=False,
    )

    updated = msa_server.preprocess_colabfold_msas(query_set, settings)
    protein, rna = updated.queries["job"].chains

    assert protein.main_msa_file_paths[0].exists()
    assert rna.main_msa_file_paths[0].exists()
    assert protein.main_msa_file_paths[0].parent == tmp_path / "main"
    assert rna.main_msa_file_paths[0].parent == tmp_path / "main"

    protein_npz = np.load(protein.main_msa_file_paths[0], allow_pickle=True)
    rna_npz = np.load(rna.main_msa_file_paths[0], allow_pickle=True)
    assert "colabfold_main" in protein_npz
    assert "riboseek_main" in rna_npz


def test_preprocess_colabfold_msas_falls_back_on_riboseek_timeout(
    tmp_path, monkeypatch
):
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "job": {
                    "chains": [
                        {
                            "molecule_type": "rna",
                            "chain_ids": ["A"],
                            "sequence": "GCGG",
                        },
                    ],
                }
            }
        }
    )

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("Riboseek timed out")

    monkeypatch.setattr(msa_server, "query_riboseek_msa_server", raise_timeout)

    settings = msa_server.MsaComputationSettings(
        msa_output_directory=tmp_path,
        msa_file_format="a3m",
        save_mappings=False,
    )

    updated = msa_server.preprocess_colabfold_msas(query_set, settings)
    rna = updated.queries["job"].chains[0]

    assert rna.main_msa_file_paths[0].exists()
    assert rna.main_msa_file_paths[0].read_text() == ">query\nGCGG\n"


def test_msa_settings_include_riboseek_main():
    settings = MSASettings()

    assert settings.max_seq_counts["riboseek_main"] == 16384
    assert "riboseek_main" in settings.aln_order
