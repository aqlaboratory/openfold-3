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

"""Tests for the ColabFold MSA server module."""

import getpass
import io
import json
import shutil
import tarfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic_core import Url

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.core.data.io.sequence.msa import parse_a3m as parse_msa_a3m
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.tools.colabfold_msa_server import (
    ColabFoldQueryRunner,
    ColabFoldServerResultError,
    ComplexGroup,
    MsaComputationSettings,
    add_msa_paths_to_iqs,
    augment_main_msa_with_query_sequence,
    collect_colabfold_msa_data,
    get_sequence_hash,
    preprocess_colabfold_msas,
    query_colabfold_msa_server,
    remap_colabfold_template_chain_ids,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    InferenceJobConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)

_MOCK_FETCH_TARGET = (
    "openfold3.core.data.tools.colabfold_msa_server.fetch_label_to_author_chain_ids"
)
_MOCK_QUERY_TARGET = (
    "openfold3.core.data.tools.colabfold_msa_server.query_colabfold_msa_server"
)

# Realistic label->author mappings for test PDB entries.
# 1RNB: label B -> author A (protein), label A -> author C (DNA)
# 4PQX: identity mapping
_MOCK_LABEL_TO_AUTHOR = {
    "1rnb": {"A": "C", "B": "A"},
    "4pqx": {"A": "A"},
    "test": {"A": "A", "B": "B", "C": "C"},
}


def _mock_fetch_label_to_author(pdb_ids):
    """Return mock label->author mappings for known test PDB IDs."""
    return {pid: _MOCK_LABEL_TO_AUTHOR.get(pid, {}) for pid in pdb_ids}


def _make_m8_dataframe(template_ids: list[str], m_index: int = 101) -> pd.DataFrame:
    """Build a minimal m8-format DataFrame for testing.

    See docs/source/template_how_to.md § 1.1.3 for the m8 column spec.
    """
    n = len(template_ids)
    return pd.DataFrame(
        {
            0: [m_index] * n,
            1: template_ids,
            2: [0.98] * n,
            3: [100] * n,
            4: [1] * n,
            5: [0] * n,
            6: [1] * n,
            7: [100] * n,
            8: [1] * n,
            9: [100] * n,
            10: [1e-10] * n,
            11: [100] * n,
            12: ["100M"] * n,
        }
    )


@pytest.fixture
def multimer_query_set():
    return InferenceQuerySet.model_validate(
        {
            "queries": {
                "query1": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A", "C"],
                            "sequence": "SHORTDUMMYSEQ",
                        },
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["B", "D"],
                            "sequence": "LONGERDUMMYSEQUENCE",
                        },
                    ]
                }
            }
        }
    )


@pytest.fixture
def multimer_sequences(multimer_query_set):
    return [c.sequence for c in multimer_query_set.queries["query1"].chains]


class TestColabfoldMapping:
    def test_colabfold_mapping_on_multimer_query(
        self, multimer_query_set, multimer_sequences
    ):
        """Test that colabfold mapper contents for a multimer query."""
        mapper = collect_colabfold_msa_data(inference_query_set=multimer_query_set)
        assert len(mapper.rep_id_to_seq) == 2, "Expected 2 unique sequences"

        expected_sequences = multimer_sequences
        complex_group = mapper.complex_id_to_complex_group.values()
        assert set(*complex_group) == set(expected_sequences), (
            "Expected complex group sequences to match the query chains"
        )

    def test_complex_id_same_on_permutation_of_sequences(self):
        order1 = ["AAAA", "BBBB"]
        order2 = ["BBBB", "AAAA"]
        assert ComplexGroup(order1).rep_id == ComplexGroup(order2).rep_id


class TestRemapColabfoldTemplateChainIds:
    """Tests for remap_colabfold_template_chain_ids (RCSB calls mocked)."""

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    def test_remap_author_to_label(self, _mock_fetch):
        """1rnb_A (author) should be remapped to 1rnb_B (label)."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_A", "4pqx_A"]),
            m_with_templates={101},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert "rep1" in result
        remapped_ids = result["rep1"][1].tolist()
        assert remapped_ids[0] == "1rnb_B"
        assert remapped_ids[1] == "4pqx_A"

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    def test_unknown_author_chain_skipped(self, _mock_fetch):
        """When the author chain ID isn't in the API response, skip that template."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_Z"]),
            m_with_templates={101},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert "rep1" in result
        # The template with unmappable chain Z should be dropped
        assert len(result["rep1"]) == 0

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    def test_unknown_chain_drops_only_bad_rows(self, _mock_fetch):
        """Valid templates are kept; only unmappable ones are dropped."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_A", "1rnb_Z", "4pqx_A"]),
            m_with_templates={101},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert "rep1" in result
        remapped_ids = result["rep1"][1].tolist()
        # 1rnb_Z dropped, 1rnb_A remapped to 1rnb_B, 4pqx_A kept
        assert len(remapped_ids) == 2
        assert remapped_ids[0] == "1rnb_B"
        assert remapped_ids[1] == "4pqx_A"

    def test_skips_rep_without_templates(self):
        """Rep IDs not in m_with_templates should be skipped (no fetch needed)."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_A"]),
            m_with_templates={999},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert len(result) == 0


class TestColabFoldQueryRunner:
    def _construct_monomer_query(self, sequence):
        return InferenceQuerySet.model_validate(
            {
                "queries": {
                    "query1": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": sequence,
                            }
                        ]
                    }
                }
            }
        )

    @staticmethod
    def _construct_dummy_a3m(seqs, **unused_kwargs):
        result = [
            textwrap.dedent(
                f"""
            >101
            {seq}
            >seq2
            {"A" * len(seq)}
            >seq3
            {"B" * len(seq)}
            """
            )
            for seq in seqs
        ]
        return result

    @staticmethod
    def _construct_dummy_a3m_with_raw_output(seqs, prefix, **unused_kwargs):
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / "server-output.a3m").write_text("raw output")
        if prefix.name == "main":
            (prefix / "pdb70.m8").touch()
        raw_output_callback = unused_kwargs.get("raw_output_callback")
        if raw_output_callback is not None:
            raw_output_callback(prefix)
        return TestColabFoldQueryRunner._construct_dummy_a3m(seqs)

    @staticmethod
    def _make_dummy_template_file(path: Path):
        raw_main_dir = path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {0: [101, 101, 102], 1: ["test_A", "test_B", "test_C"], 2: [0, 1, 2]}
        ).to_csv(raw_main_dir / "pdb70.m8", header=False, index=False, sep="\t")

    @staticmethod
    def _make_empty_template_file(path: Path):
        """Create an empty pdb70.m8 file to simulate ColabFold empty templates."""
        raw_main_dir = path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        # Create an empty file (0 bytes)
        (raw_main_dir / "pdb70.m8").touch()

    @patch("openfold3.core.data.tools.colabfold_msa_server.requests.post")
    def test_submit_url_has_no_double_slash_with_url_host(self, mock_post, tmp_path):
        """Regression: a pydantic Url host (trailing slash) must not yield
        '...com//ticket/msa'. The server 301-redirects the doubled slash, which
        makes requests downgrade the POST to GET and drop the body, so the
        server returns 'invalid ID' and the run fails with a misleading error.
        """
        # Make submit() return ERROR so the function exits right after the POST,
        # before any polling/download -- we only care about the URL that was built.
        mock_post.return_value.json.return_value = {"status": "ERROR"}

        with pytest.raises(Exception, match="MMseqs2 API is giving errors"):
            query_colabfold_msa_server(
                ["TESTSEQ"],
                prefix=tmp_path / "raw",
                user_agent="test-agent",
                host_url=Url("https://api.colabfold.com"),  # str() -> trailing slash
            )

        called_url = mock_post.call_args.args[0]
        assert called_url == "https://api.colabfold.com/ticket/msa", called_url
        assert "//" not in called_url.split("://", 1)[1], "doubled slash in path"

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    @patch(_MOCK_QUERY_TARGET)
    def test_runner_on_multimer_example(
        self,
        mock_query,
        _mock_chain_map,
        tmp_path,
        multimer_query_set,
        multimer_sequences,
    ):
        # dummy a3m output
        mock_query.return_value = [">seq1\nAAA\n", ">seq2\nBBBBB\n"]
        self._make_dummy_template_file(tmp_path)

        mapper = collect_colabfold_msa_data(multimer_query_set)
        runner = ColabFoldQueryRunner(
            colabfold_mapper=mapper,
            output_directory=tmp_path,
            msa_file_format="npz",
            user_agent="test-agent",
            host_url="https://dummy.url",
        )

        runner.query_format_main()
        runner.query_format_paired()
        expected_unpaired_dir = tmp_path / "main"
        assert expected_unpaired_dir.exists()

        multimer_complex_group = ComplexGroup(multimer_sequences)
        expected_paired_dir = tmp_path / f"paired/{multimer_complex_group.rep_id}"
        assert expected_paired_dir.exists()

        expected_files = [f"{get_sequence_hash(s)}.npz" for s in multimer_sequences]
        for f in expected_files:
            assert (expected_unpaired_dir / f).exists()
            assert (expected_paired_dir / f).exists()

    @patch(_MOCK_QUERY_TARGET)
    def test_raw_output_is_saved_before_formatting(
        self, mock_query, tmp_path, multimer_query_set
    ):
        mock_query.side_effect = self._construct_dummy_a3m_with_raw_output
        settings = MsaComputationSettings(
            msa_file_format="npz",
            colabfold_output_dir=tmp_path / "raw-records",
        )
        settings.set_saved_output_root(tmp_path / "saved")
        mapper = collect_colabfold_msa_data(multimer_query_set)
        paired_id = next(iter(mapper.complex_id_to_complex_group))
        saved_output = settings.saved_output_directory
        saved_raw = settings.saved_colabfold_output_directory
        saved_paired_raw = saved_raw / f"paired/{paired_id}/server-output.a3m"

        def fail_during_paired_formatting(alignment):
            if saved_paired_raw.exists():
                raise RuntimeError("formatting failed")
            return parse_msa_a3m(alignment)

        with (
            patch(
                "openfold3.core.data.tools.colabfold_msa_server.parse_a3m",
                side_effect=fail_during_paired_formatting,
            ),
            pytest.raises(RuntimeError, match="formatting failed"),
        ):
            preprocess_colabfold_msas(multimer_query_set, settings)

        assert (saved_raw / "main/server-output.a3m").exists()
        assert saved_paired_raw.exists()
        assert not saved_output.exists()
        settings.cleanup_workspace()

    @patch(_MOCK_QUERY_TARGET)
    @pytest.mark.parametrize(
        "save_openfold,save_colabfold",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_saved_output_options_are_independent(
        self, mock_query, tmp_path, save_openfold, save_colabfold
    ):
        mock_query.side_effect = self._construct_dummy_a3m_with_raw_output
        saved_output_root = tmp_path / "saved"
        settings = MsaComputationSettings(
            msa_file_format="a3m",
            save_openfold_outputs=save_openfold,
            save_colabfold_outputs=save_colabfold,
        )
        settings.set_saved_output_root(saved_output_root)
        workspace = settings.workspace_directory
        saved_output = settings.saved_output_directory

        query = self._construct_monomer_query("TEST")
        processed = preprocess_colabfold_msas(query, settings)
        saved_main = (
            saved_output / "main" / get_sequence_hash("TEST") / "colabfold_main.a3m"
        )
        saved_raw = settings.saved_colabfold_output_directory / "main/server-output.a3m"

        settings.cleanup_workspace()

        assert saved_main.exists() is save_openfold
        assert saved_raw.exists() is save_colabfold
        assert (saved_output / "mappings/seq_to_rep_id.json").exists() is save_openfold
        expected_root = saved_output if save_openfold else workspace
        assert (
            processed.queries["query1"]
            .chains[0]
            .main_msa_file_paths[0]
            .is_relative_to(expected_root)
        )

    @patch(_MOCK_QUERY_TARGET)
    def test_explicit_output_directory_survives_workspace_cleanup(
        self, mock_query, tmp_path
    ):
        mock_query.side_effect = self._construct_dummy_a3m_with_raw_output
        output_directory = tmp_path / "colabfold_msas"
        output_directory.mkdir()
        sentinel = output_directory / "existing.txt"
        sentinel.write_text("keep")
        settings = MsaComputationSettings(
            msa_file_format="a3m",
            msa_output_directory=output_directory,
            cleanup_msa_dir=False,
        )

        query = self._construct_monomer_query("TEST")
        processed = preprocess_colabfold_msas(query, settings)
        settings.cleanup_workspace()

        saved_msa = (
            output_directory / "main" / get_sequence_hash("TEST") / "colabfold_main.a3m"
        )
        assert sentinel.read_text() == "keep"
        assert saved_msa.exists()
        assert (output_directory / "raw/main/server-output.a3m").exists()
        assert processed.queries["query1"].chains[0].main_msa_file_paths == [saved_msa]

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    @patch(_MOCK_QUERY_TARGET, side_effect=_construct_dummy_a3m)
    @pytest.mark.parametrize(
        "msa_file_format", ["a3m", "npz"], ids=lambda fmt: f"format={fmt}"
    )
    def test_msa_generation_on_multiple_queries_with_same_name(
        self,
        mock_query,
        _mock_chain_map,
        tmp_path,
        msa_file_format,
    ):
        test_sequences = ["TEST", "LONGERTEST"]

        # dummy tsv output
        self._make_dummy_template_file(tmp_path)

        # run a separate query with the same name for each test sequence
        for sequence in test_sequences:
            query = self._construct_monomer_query(sequence)
            mapper = collect_colabfold_msa_data(query)
            runner = ColabFoldQueryRunner(
                colabfold_mapper=mapper,
                output_directory=tmp_path,
                msa_file_format=msa_file_format,
                user_agent="test-agent",
                host_url="https://dummy.url",
            )
            runner.query_format_main()

        match msa_file_format:
            case "a3m":
                expected_files = [
                    f"{get_sequence_hash(s)}/colabfold_main.a3m" for s in test_sequences
                ]
            case "npz":
                expected_files = [f"{get_sequence_hash(s)}.npz" for s in test_sequences]

        for f in expected_files:
            assert (tmp_path / "main" / f).exists(), (
                f"Expected file {f} not found in main directory"
            )

    @pytest.mark.parametrize(
        "msa_file_format", ["a3m", "npz"], ids=lambda fmt: f"format={fmt}"
    )
    def test_augment_main_msa_with_query_sequence(
        self,
        tmp_path,
        msa_file_format,
    ):
        sequence = "TEST"
        msa_compute_settings = MsaComputationSettings(
            msa_file_format=msa_file_format,
            server_user_agent="test-agent",
            server_url="https://dummy.url",
            save_mappings=True,
        )
        msa_compute_settings.set_saved_output_root(tmp_path / "saved-openfold")

        query = self._construct_monomer_query(sequence)
        augmented = augment_main_msa_with_query_sequence(query, msa_compute_settings)
        match msa_file_format:
            case "a3m":
                f = f"{get_sequence_hash(sequence)}/colabfold_main.a3m"
            case "npz":
                f = f"{get_sequence_hash(sequence)}.npz"

        expected_file = msa_compute_settings.saved_output_directory / "dummy" / f
        assert expected_file.exists(), f"Expected file {f} not found in main directory"

        paths_in_augmented = augmented.queries["query1"].chains[0].main_msa_file_paths
        assert len(paths_in_augmented) == 1
        assert expected_file == paths_in_augmented[0], (
            f"Unexpected MSA path in augmented query set: {paths_in_augmented[0]}"
        )
        msa_compute_settings.cleanup_workspace()
        assert expected_file.exists()

    @patch(_MOCK_FETCH_TARGET, side_effect=_mock_fetch_label_to_author)
    @patch(_MOCK_QUERY_TARGET)
    @pytest.mark.parametrize(
        "msa_file_format", ["a3m", "npz"], ids=lambda fmt: f"{fmt}"
    )
    def test_features_on_multiple_queries_with_same_name(
        self,
        mock_query,
        _mock_chain_map,
        tmp_path,
        msa_file_format,
    ):
        """Integration test for making predictions with fake MSA data."""
        mock_query.side_effect = self._construct_dummy_a3m_with_raw_output
        test_sequences = ["TEST", "LONGERTEST"]

        for sequence in test_sequences:
            query_set = self._construct_monomer_query(sequence)
            msa_compute_settings = MsaComputationSettings(
                msa_file_format=msa_file_format,
                server_user_agent="test-agent",
                server_url="https://dummy.url",
                save_mappings=True,
                msa_output_directory=tmp_path,
                cleanup_msa_dir=False,
            )
            query_set = preprocess_colabfold_msas(
                inference_query_set=query_set, compute_settings=msa_compute_settings
            )
            inference_config = InferenceJobConfig(
                query_set=query_set,
                msa=MSASettings(max_seq_counts={"colabfold_main": 10}),
                template_preprocessor_settings=TemplatePreprocessorSettings(),
            )
            inference_spec = InferenceDatasetSpec(config=inference_config)

            data_config = DataModuleConfig(
                datasets=[inference_spec],
                batch_size=1,
                epoch_len=1,
                num_epochs=1,
            )

            data_module = DataModule(data_config)

            data_module.setup()
            dataloader = data_module.predict_dataloader()

            expected_msa = 4  # based on _construct_dummy_a3m
            expected_shape = (1, expected_msa, len(sequence), 32)
            # the implicit iter here is causing a segfault in Python 3.13
            for batch in dataloader:
                b, s, t, e = batch["msa"].shape
                b_expected, s_expected, t_expected, e_expected = expected_shape
                assert b == b_expected, f"Batch size mismatch: {b} != {b_expected}"
                assert t == t_expected, f"Target length mismatch: {t} != {t_expected}"
                assert e == e_expected, f"Feature size mismatch: {e} != {e_expected}"

            msa_compute_settings.cleanup_workspace()

        with open(tmp_path / "mappings/seq_to_rep_id.json") as f:
            assert set(json.load(f)) == set(test_sequences)
        assert (tmp_path / "raw/main").is_dir()

    @patch(_MOCK_QUERY_TARGET, side_effect=_construct_dummy_a3m)
    def test_empty_m8_file_handling(
        self,
        mock_query,
        tmp_path,
    ):
        """Test that empty pdb70.m8 file is handled gracefully without crashing.
        Runs logic in `preprocess_colabfold_msas` manually in order to add assertions within the run.
        """
        test_sequence = "TESTSEQUENCE"
        query = self._construct_monomer_query(test_sequence)

        self._make_empty_template_file(tmp_path)

        mapper = collect_colabfold_msa_data(query)
        runner = ColabFoldQueryRunner(
            colabfold_mapper=mapper,
            output_directory=tmp_path,
            msa_file_format="npz",
            user_agent="test-agent",
            host_url="https://dummy.url",
        )

        # Should not raise EmptyDataError or any other exception
        runner.query_format_main()

        # Verify MSA processing still works
        expected_unpaired_dir = tmp_path / "main"
        assert expected_unpaired_dir.exists(), "Expected main MSA directory to exist"

        expected_file = f"{get_sequence_hash(test_sequence)}.npz"
        assert (expected_unpaired_dir / expected_file).exists(), (
            f"Expected MSA file {expected_file} to exist"
        )

        # Verify no template files are created (since m8 file is empty)
        template_alignments_dir = tmp_path / "template"
        if template_alignments_dir.exists():
            # If directory exists, it should be empty (no template files created)
            template_files = list(template_alignments_dir.rglob("*.m8"))
            assert len(template_files) == 0, (
                "Expected no template files to be created when m8 file is empty"
            )

        # Continue with inner loop of `preprocess_colabfold_msas` to test template chain assignment
        processed_query_set = add_msa_paths_to_iqs(
            inference_query_set=query,
            colabfold_mapper=mapper,
            output_directory=tmp_path,
        )

        # Verify that template fields are None/empty for all chains
        for query_name, query_obj in processed_query_set.queries.items():
            for chain in query_obj.chains:
                assert chain.template_alignment_file_path is None, (
                    f"Expected template_alignment_file_path to be None for chain "
                    f"{chain.chain_ids} of query {query_name} when template file "
                    f"is empty, but got {chain.template_alignment_file_path}"
                )
                assert chain.template_entry_chain_ids is None, (
                    f"Expected template_entry_chain_ids to be None for chain "
                    f"{chain.chain_ids} of query {query_name} when template file"
                    f"is empty, but got {chain.template_entry_chain_ids}"
                )

    @patch(_MOCK_QUERY_TARGET)
    def test_preprocess_rejects_an_existing_workspace(self, mock_query, tmp_path):
        query = self._construct_monomer_query("TESTSEQUENCE")
        msa_compute_settings = MsaComputationSettings(
            msa_file_format="npz",
            server_user_agent="test-agent",
            server_url="https://dummy.url",
        )
        workspace = msa_compute_settings.workspace_directory
        workspace.mkdir(parents=True)
        sentinel = workspace / "keep.txt"
        sentinel.write_text("keep")

        with pytest.raises(FileExistsError):
            preprocess_colabfold_msas(
                inference_query_set=query, compute_settings=msa_compute_settings
            )

        mock_query.assert_not_called()
        assert not (workspace / "mappings").exists()
        msa_compute_settings.cleanup_workspace()
        assert sentinel.exists()
        shutil.rmtree(workspace)


class _ValidationCase(NamedTuple):
    """A bad ColabFold download and the error it should trigger (issue #269)."""

    members: dict[str, bytes]  # tarball contents (name -> bytes)
    use_pairing: bool
    match: str  # substring expected in the raised error message


# The server returned the wrong/incomplete job: the expected a3m file is absent or
# empty in the downloaded tarball.
_VALIDATION_CASES = [
    pytest.param(
        _ValidationCase(
            members={
                "uniref.a3m": b">101\nAAAA\n",
                "bfd.mgnify30.metaeuk30.smag30.a3m": b">101\nAAAA\n",
                "pdb70.m8": b"",
            },
            use_pairing=True,
            match="pair.a3m",
        ),
        id="paired_gets_unpaired_tarball",
    ),
    pytest.param(
        _ValidationCase(
            members={"pdb70.m8": b"templates\n"},
            use_pairing=False,
            match="uniref.a3m",
        ),
        id="unpaired_missing_uniref",
    ),
    pytest.param(
        _ValidationCase(
            members={"pair.a3m": b""},
            use_pairing=True,
            match="pair.a3m",
        ),
        id="empty_pair_a3m",
    ),
]


class TestQueryColabFoldServerValidation:
    """Regression tests for issue #269.

    The ColabFold server can return the wrong cached job for a ticket (e.g. an
    unpaired MSA -- no ``pair.a3m`` -- for a paired request). ``query_colabfold_msa_
    server`` must reject such a download with a clear ``ColabFoldServerResultError``
    instead of crashing later on a bare ``FileNotFoundError``.
    """

    @staticmethod
    def _write_tarball(out_tar_gz: Path, members: dict[str, bytes]) -> None:
        """Write a gzipped tar of ``members`` (name -> bytes) at ``out_tar_gz``."""
        with tarfile.open(out_tar_gz, "w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    @pytest.mark.parametrize("case", _VALIDATION_CASES)
    def test_rejects_unexpected_download(
        self, case: _ValidationCase, tmp_path: Path
    ) -> None:
        """A download missing an expected a3m file raises ColabFoldServerResultError.

        Pre-creating ``out.tar.gz`` makes ``query_colabfold_msa_server`` skip the
        submit/download (no network) and go straight to extraction + validation.
        """
        self._write_tarball(tmp_path / "out.tar.gz", case.members)
        callback_paths: list[Path] = []

        with pytest.raises(ColabFoldServerResultError, match=case.match):
            query_colabfold_msa_server(
                ["AAAA", "CCCC"],
                prefix=tmp_path,
                user_agent="test-agent",
                use_pairing=case.use_pairing,
                raw_output_callback=callback_paths.append,
            )

        assert callback_paths == []

    def test_valid_paired_download_passes(self, tmp_path: Path) -> None:
        """A paired download containing a valid pair.a3m returns the alignments."""
        self._write_tarball(
            tmp_path / "out.tar.gz",
            {"pair.a3m": b">101\nAAAA\n\x00>102\nCCCC\n", "pair.sh": b"#\n"},
        )
        callback_paths: list[Path] = []

        result = query_colabfold_msa_server(
            ["AAAA", "CCCC"],
            prefix=tmp_path,
            user_agent="test-agent",
            use_pairing=True,
            raw_output_callback=callback_paths.append,
        )

        assert len(result) == 2
        assert callback_paths == [tmp_path]


class TestRemapObsoletePdb:
    """Regression test for GitHub issue #170.

    When ColabFold returns a template hit for an obsolete PDB (e.g. 7QE7),
    the RCSB API returns no chain mapping.  The function should warn and
    fall back to using the author chain ID as the label chain ID, rather
    than crashing the entire run.
    """

    @staticmethod
    def _mock_fetch_excluding_obsolete(pdb_ids):
        """Simulate RCSB not returning data for obsolete PDB entries."""
        known = {
            "4pqx": {"A": "A"},
        }
        # Obsolete PDB "7qe7" is intentionally absent from known
        return {pid: known[pid] for pid in pdb_ids if pid in known}

    @patch(_MOCK_FETCH_TARGET)
    def test_obsolete_pdb_falls_back(self, mock_fetch):
        """Obsolete PDB with no RCSB mapping falls back to author chain ID."""
        mock_fetch.side_effect = self._mock_fetch_excluding_obsolete

        # Template hits include an obsolete PDB entry (7qe7)
        df = _make_m8_dataframe(["7qe7_A", "4pqx_A"])

        result = remap_colabfold_template_chain_ids(
            template_alignments=df,
            m_with_templates={101},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        remapped_ids = result["rep1"][1].tolist()
        # Obsolete entry falls back to author chain ID
        assert remapped_ids[0] == "7qe7_A"
        # Non-obsolete entry is remapped normally
        assert remapped_ids[1] == "4pqx_A"


class TestMsaComputationSettings:
    @pytest.mark.parametrize("cleanup_msa_dir", [False, True])
    def test_workspace_cleanup_is_unconditional(self, cleanup_msa_dir):
        settings = MsaComputationSettings(cleanup_msa_dir=cleanup_msa_dir)
        settings.create_workspace()

        settings.cleanup_workspace()

        assert not settings.workspace_directory.exists()

    def test_workspace_cleanup_raises_and_can_retry_after_failed_removal(self):
        settings = MsaComputationSettings()
        settings.create_workspace()

        with (
            patch("shutil.rmtree", side_effect=OSError("failed")),
            pytest.raises(OSError, match="failed"),
        ):
            settings.cleanup_workspace()

        assert settings.workspace_directory.exists()
        settings.cleanup_workspace()
        assert not settings.workspace_directory.exists()

    def test_cli_output_dir_is_persistent(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
            msa_file_format: a3m
            server_user_agent: test-agent
            server_url: https://dummy.url
            save_openfold_outputs: false
        """)
        cli_output_dir = tmp_path / "cli_dir"
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        msa_settings = MsaComputationSettings.from_config_with_cli_override(
            cli_output_dir, test_yaml_file
        )
        assert msa_settings.save_openfold_outputs
        assert msa_settings.save_colabfold_outputs
        assert msa_settings.saved_output_directory == cli_output_dir
        assert msa_settings.saved_colabfold_output_directory == cli_output_dir / "raw"
        assert msa_settings.workspace_directory != cli_output_dir

    def test_cli_rejects_a_different_configured_output_directory(self, tmp_path):
        config_file = tmp_path / "runner.yml"
        config_file.write_text(
            f"msa_output_directory: {tmp_path / 'configured-output'}\n"
        )

        with pytest.raises(ValueError, match="Output directory mismatch"):
            MsaComputationSettings.from_config_with_cli_override(
                tmp_path / "cli-output", config_file
            )

    def test_msa_settings_keep_output_state_separate_with_readable_run_names(
        self, tmp_path
    ):
        output_directory = tmp_path / "saved"
        fixed_time = datetime.fromisoformat("2026-08-06T12:00:00+00:00")
        with (
            patch(
                "openfold3.core.data.tools.colabfold_msa_server.datetime"
            ) as mock_datetime,
            patch(
                "openfold3.core.data.tools.colabfold_msa_server.secrets.token_hex",
                side_effect=["12345678", "abcdef01"],
            ),
        ):
            mock_datetime.now.return_value = fixed_time
            settings = MsaComputationSettings(msa_output_directory=output_directory)
            other_settings = MsaComputationSettings()
        run_name_prefix = f"msa-{getpass.getuser()}-"

        assert settings.saved_output_directory == output_directory
        assert settings.saved_colabfold_output_directory == output_directory / "raw"
        assert settings.workspace_directory != output_directory
        assert other_settings.saved_output_directory is None
        for run_name in (
            settings.run_directory_name,
            other_settings.run_directory_name,
        ):
            assert run_name.startswith(run_name_prefix)
            timestamp, random_suffix = run_name.removeprefix(run_name_prefix).rsplit(
                "-", 1
            )
            datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
            assert len(random_suffix) == 8
            int(random_suffix, 16)
        assert other_settings.workspace_directory != settings.workspace_directory

    @pytest.mark.parametrize("nested", [False, True])
    def test_workspace_cannot_contain_openfold_output(self, nested):
        settings = MsaComputationSettings(save_colabfold_outputs=False)
        settings.msa_output_directory = (
            settings.workspace_directory / "saved"
            if nested
            else settings.workspace_directory
        )

        with pytest.raises(ValueError, match="must not overlap"):
            settings.validate_output_paths()

        assert not settings.workspace_directory.exists()

    @pytest.mark.parametrize("nested", [False, True])
    def test_workspace_cannot_contain_colabfold_output(self, nested):
        settings = MsaComputationSettings(save_openfold_outputs=False)
        settings.colabfold_output_dir = (
            settings.workspace_directory
            if nested
            else settings.workspace_directory.parent
        )

        with pytest.raises(ValueError, match="must not overlap"):
            settings.validate_output_paths()

        assert not settings.workspace_directory.exists()
