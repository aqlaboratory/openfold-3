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

"""Tests to check handling of colabofold MSA data."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.tools.colabfold_msa_server import (
    ColabFoldQueryRunner,
    ComplexGroup,
    MsaComputationSettings,
    collect_colabfold_msa_data,
    get_sequence_hash,
    preprocess_colabfold_msas,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    InferenceJobConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
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
    def _make_dummy_template_file(path: Path):
        raw_main_dir = path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {0: [101, 101, 102], 1: ["test_A", "test_B", "test_C"], 2: [0, 1, 2]}
        ).to_csv(raw_main_dir / "pdb70.m8", header=False, index=False, sep="\t")

    @staticmethod
    def _make_empty_template_file(path: Path):
        """Create an empty pdb70.m8 file to simulate ColabFold returning empty template file."""
        raw_main_dir = path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        # Create an empty file (0 bytes)
        (raw_main_dir / "pdb70.m8").touch()

    @patch("openfold3.core.data.tools.colabfold_msa_server.query_colabfold_msa_server")
    def test_runner_on_multimer_example(
        self,
        mock_query,
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

    @patch(
        "openfold3.core.data.tools.colabfold_msa_server.query_colabfold_msa_server",
        side_effect=_construct_dummy_a3m,
    )
    @pytest.mark.parametrize(
        "msa_file_format", ["a3m", "npz"], ids=lambda fmt: f"format={fmt}"
    )
    def test_msa_generation_on_multiple_queries_with_same_name(
        self,
        mock_query,
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

    @patch(
        "openfold3.core.data.tools.colabfold_msa_server.query_colabfold_msa_server",
        side_effect=_construct_dummy_a3m,
    )
    @pytest.mark.parametrize(
        "msa_file_format", ["a3m", "npz"], ids=lambda fmt: f"{fmt}"
    )
    def test_features_on_multiple_queries_with_same_name(
        self,
        mock_query,
        tmp_path,
        msa_file_format,
    ):
        """Integration test for making predictions with fake MSA data."""
        test_sequences = ["TEST", "LONGERTEST"]

        for sequence in test_sequences:
            # dummy tsv output
            query_set = self._construct_monomer_query(sequence)
            self._make_dummy_template_file(tmp_path)
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
                assert batch["msa"].shape == expected_shape

        # Test contents of mapping file after all runs
        with open(tmp_path / "mappings/seq_to_rep_id.json") as f:
            assert set(json.load(f).keys()) == set(test_sequences), (
                "Expected all test sequences to be present in the mapping file"
            )

    @patch(
        "openfold3.core.data.tools.colabfold_msa_server.query_colabfold_msa_server",
        side_effect=_construct_dummy_a3m,
    )
    def test_empty_m8_file_handling(
        self,
        mock_query,
        tmp_path,
    ):
        """Test that empty pdb70.m8 file is handled gracefully without crashing."""
        test_sequence = "TESTSEQUENCE"
        query = self._construct_monomer_query(test_sequence)
        
        # Create an empty pdb70.m8 file (0 bytes) to simulate ColabFold returning empty template file
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
        
        # Test preprocess_colabfold_msas with empty template file
        msa_compute_settings = MsaComputationSettings(
            msa_file_format="npz",
            server_user_agent="test-agent",
            server_url="https://dummy.url",
            save_mappings=True,
            msa_output_directory=tmp_path,
            cleanup_msa_dir=False,
        )
        
        # Call preprocess_colabfold_msas - should not raise any exception
        processed_query_set = preprocess_colabfold_msas(
            inference_query_set=query,
            compute_settings=msa_compute_settings
        )
        
        # Verify that template fields are None/empty for all chains
        for query_name, query_obj in processed_query_set.queries.items():
            for chain in query_obj.chains:
                assert chain.template_alignment_file_path is None, (
                    f"Expected template_alignment_file_path to be None for chain "
                    f"{chain.chain_ids} of query {query_name} when template file is empty, "
                    f"but got {chain.template_alignment_file_path}"
                )
                assert chain.template_entry_chain_ids is None, (
                    f"Expected template_entry_chain_ids to be None for chain "
                    f"{chain.chain_ids} of query {query_name} when template file is empty, "
                    f"but got {chain.template_entry_chain_ids}"
                )


class TestMsaComputationSettings:
    def test_cli_output_dir_overrides_config(self, tmp_path):
        """Test that CLI output directory overrides config file setting."""
        test_yaml_str = textwrap.dedent("""\
            msa_file_format: a3m 
            server_user_agent: test-agent
            server_url: https://dummy.url
        """)
        cli_output_dir = tmp_path / "cli_dir"
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        msa_settings = MsaComputationSettings.from_config_with_cli_override(
            cli_output_dir, test_yaml_file
        )

        assert Path(msa_settings.msa_output_directory) == cli_output_dir, (
            "Expected CLI output directory to override default settings"
        )

    def test_cli_output_dir_conflict_raises(self, tmp_path):
        """Test that conflict between CLI and config output dirs raises ValueError."""
        test_yaml_str = textwrap.dedent(f"""\
            msa_file_format: a3m 
            msa_output_directory: {tmp_path / "other_dir"}
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        cli_output_dir = tmp_path / "cli_dir"

        with pytest.raises(ValueError) as exc_info:
            MsaComputationSettings.from_config_with_cli_override(
                cli_output_dir, test_yaml_file
            )

        assert "Output directory mismatch" in str(exc_info.value), (
            "Expected ValueError on output directory conflict"
        )
