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
    _parse_a3m_file_by_m,
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


class TestParseA3mFileByM:
    """Test suite for _parse_a3m_file_by_m function."""

    def test_parse_single_m_value(self, tmp_path):
        """Test parsing a3m file with a single M value."""
        a3m_content = textwrap.dedent(
            """\
            >101
            SEQUENCE1
            >UniRef100_A0A123
            MATCH1
            MATCH1CONT
            >UniRef100_B0B456
            MATCH2
            """
        )
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_text(a3m_content)

        result = _parse_a3m_file_by_m(a3m_file)

        assert 101 in result, "Expected M value 101 in result"
        assert len(result) == 1, "Expected only one M value"
        lines = result[101]
        assert len(lines) == 7, "Expected 7 lines for M=101 (M header, sequence, 2 UniRef headers, 3 match lines)"
        assert lines[0] == ">101\n", "First line should be M header"
        assert lines[1] == "SEQUENCE1\n", "Second line should be sequence"
        assert ">UniRef100_A0A123\n" in lines, "Should contain UniRef header"
        assert "MATCH1\n" in lines, "Should contain match sequence"
        assert "MATCH1CONT\n" in lines, "Should contain continuation of match sequence"
        assert ">UniRef100_B0B456\n" in lines, "Should contain second UniRef header"
        assert "MATCH2\n" in lines, "Should contain second match sequence"

    def test_parse_multiple_m_values(self, tmp_path):
        """Test parsing a3m file with multiple M values separated by null bytes."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\n>UniRef100_A0A123\nMATCH1\n\x00>102\nSEQUENCE2\n>UniRef100_B0B456\nMATCH2\n\x00>103\nSEQUENCE3\n>UniRef100_C0C789\nMATCH3\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 3, "Expected 3 M values"
        assert 101 in result, "Expected M value 101"
        assert 102 in result, "Expected M value 102"
        assert 103 in result, "Expected M value 103"

        # Check M=101 section
        lines_101 = result[101]
        assert lines_101[0] == ">101\n", "M=101 should start with >101"
        assert "SEQUENCE1\n" in lines_101, "M=101 should contain SEQUENCE1"
        assert ">UniRef100_A0A123\n" in lines_101, "M=101 should contain UniRef header"
        assert ">102\n" not in lines_101, "M=101 should not contain >102"

        # Check M=102 section
        lines_102 = result[102]
        assert lines_102[0] == ">102\n", "M=102 should start with >102"
        assert "SEQUENCE2\n" in lines_102, "M=102 should contain SEQUENCE2"
        assert ">UniRef100_B0B456\n" in lines_102, "M=102 should contain UniRef header"
        assert ">103\n" not in lines_102, "M=102 should not contain >103"

        # Check M=103 section
        lines_103 = result[103]
        assert lines_103[0] == ">103\n", "M=103 should start with >103"
        assert "SEQUENCE3\n" in lines_103, "M=103 should contain SEQUENCE3"

    def test_parse_with_m_filter(self, tmp_path):
        """Test parsing with M value filter."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\n>UniRef100_A0A123\nMATCH1\n\x00>102\nSEQUENCE2\n>UniRef100_B0B456\nMATCH2\n\x00>103\nSEQUENCE3\n>UniRef100_C0C789\nMATCH3\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        # Filter to only include M=101 and M=103
        m_filter = {101: "rep1", 103: "rep3"}
        result = _parse_a3m_file_by_m(a3m_file, m_filter=m_filter)

        assert len(result) == 2, "Expected 2 M values after filtering"
        assert 101 in result, "Expected M value 101"
        assert 103 in result, "Expected M value 103"
        assert 102 not in result, "M=102 should be filtered out"

        # Check that filtered sections are correct
        lines_101 = result[101]
        assert "SEQUENCE1\n" in lines_101, "M=101 should contain SEQUENCE1"
        assert "SEQUENCE2\n" not in lines_101, "M=101 should not contain SEQUENCE2"

        lines_103 = result[103]
        assert "SEQUENCE3\n" in lines_103, "M=103 should contain SEQUENCE3"
        assert "SEQUENCE2\n" not in lines_103, "M=103 should not contain SEQUENCE2"

    def test_parse_with_null_bytes(self, tmp_path):
        """Test parsing a3m file with null bytes (\\x00) that trigger M updates."""
        a3m_content = ">101\nSEQUENCE1\n\x00>102\nSEQUENCE2\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 2, "Expected 2 M values"
        assert 101 in result, "Expected M value 101"
        assert 102 in result, "Expected M value 102"

        # Check that null bytes are removed
        lines_101 = result[101]
        assert "\x00" not in "".join(lines_101), "Null bytes should be removed from M=101"
        assert "SEQUENCE1\n" in lines_101, "M=101 should contain SEQUENCE1"

        lines_102 = result[102]
        assert "\x00" not in "".join(lines_102), "Null bytes should be removed from M=102"
        assert "SEQUENCE2\n" in lines_102, "M=102 should contain SEQUENCE2"

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty a3m file."""
        a3m_file = tmp_path / "empty.a3m"
        a3m_file.write_text("")

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 0, "Expected empty result for empty file"

    def test_parse_file_with_no_m_values(self, tmp_path):
        """Test parsing a3m file with no M value headers."""
        a3m_content = textwrap.dedent(
            """\
            >UniRef100_A0A123
            MATCH1
            >UniRef100_B0B456
            MATCH2
            """
        )
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_text(a3m_content)

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 0, "Expected empty result when no M values present"

    def test_parse_file_with_m_value_less_than_101(self, tmp_path):
        """Test parsing a3m file with M value less than 101 (should still parse)."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">100\nSEQUENCE1\n>UniRef100_A0A123\nMATCH1\n\x00>101\nSEQUENCE2\n>UniRef100_B0B456\nMATCH2\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        # Should parse both M=100 and M=101 (function doesn't filter by >= 101)
        assert 100 in result, "Expected M value 100"
        assert 101 in result, "Expected M value 101"

    def test_parse_with_complex_uniref_headers(self, tmp_path):
        """Test parsing with complex UniRef headers that might look like M values."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\n>UniRef100_10566|scaffold\nMATCH1\n>UniRef100_A0A208S2R6\nMATCH2\n\x00>102\nSEQUENCE2\n>UniRef100_12345\nMATCH3\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 2, "Expected 2 M values"
        assert 101 in result, "Expected M value 101"
        assert 102 in result, "Expected M value 102"

        # Check that complex UniRef headers are included in correct sections
        lines_101 = result[101]
        assert ">UniRef100_10566|scaffold\n" in lines_101, "Complex UniRef header should be in M=101"
        assert ">UniRef100_A0A208S2R6\n" in lines_101, "UniRef header should be in M=101"
        assert ">UniRef100_12345\n" not in lines_101, "UniRef header should not be in M=101"

        lines_102 = result[102]
        assert ">UniRef100_12345\n" in lines_102, "UniRef header should be in M=102"

    def test_parse_with_whitespace_in_m_header(self, tmp_path):
        """Test parsing M headers with whitespace."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101 \nSEQUENCE1\n\x00>102\t\nSEQUENCE2\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 2, "Expected 2 M values"
        assert 101 in result, "Expected M value 101"
        assert 102 in result, "Expected M value 102"

    def test_parse_with_filter_excluding_all(self, tmp_path):
        """Test parsing with filter that excludes all M values."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\n\x00>102\nSEQUENCE2\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        # Filter that doesn't include any M values
        m_filter = {999: "dummy"}
        result = _parse_a3m_file_by_m(a3m_file, m_filter=m_filter)

        assert len(result) == 0, "Expected empty result when all M values filtered out"

    def test_parse_nonexistent_file(self, tmp_path):
        """Test parsing a non-existent file."""
        nonexistent_file = tmp_path / "nonexistent.a3m"

        result = _parse_a3m_file_by_m(nonexistent_file)

        assert len(result) == 0, "Expected empty result for non-existent file"

    def test_parse_with_multiline_sequences(self, tmp_path):
        """Test parsing with sequences that span multiple lines."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\nPART1\nPART2\n>UniRef100_A0A123\nMATCH1\nMATCH1CONT\n\x00>102\nSEQUENCE2\nPART3\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 2, "Expected 2 M values"
        lines_101 = result[101]
        assert "SEQUENCE1\n" in lines_101, "Should contain first part of sequence"
        assert "PART1\n" in lines_101, "Should contain second part"
        assert "PART2\n" in lines_101, "Should contain third part"
        assert "PART3\n" not in lines_101, "Should not contain parts from M=102"

    def test_parse_with_uniref_header_with_metadata(self, tmp_path):
        """Test parsing with UniRef headers that have metadata (e.g., >10648|Ga0318525_12252441_1|+3|11...)."""
        # Real a3m files have null bytes (\x00) before new M value headers
        a3m_content = ">101\nSEQUENCE1\n>UniRef100_A0A123\nMATCH1\n>10648|Ga0318525_12252441_1|+3|11\t57\t0.284\t3.822E-04\t143\t229\t747\t7\t91\t95\nMATCH2\n\x00>102\nSEQUENCE2\n>UniRef100_B0B456\nMATCH3\n>10566|scaffold|+1|5\t42\t0.500\t1.444E-49\t0\t121\t122\t42\t162\t163\nMATCH4\n"
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_bytes(a3m_content.encode("utf-8"))

        result = _parse_a3m_file_by_m(a3m_file)

        assert len(result) == 2, "Expected 2 M values"
        assert 101 in result, "Expected M value 101"
        assert 102 in result, "Expected M value 102"

        # Check M=101 section - should contain the UniRef header with metadata
        lines_101 = result[101]
        assert ">101\n" in lines_101, "M=101 should start with >101"
        assert "SEQUENCE1\n" in lines_101, "M=101 should contain SEQUENCE1"
        assert ">UniRef100_A0A123\n" in lines_101, "M=101 should contain simple UniRef header"
        assert ">10648|Ga0318525_12252441_1|+3|11" in "".join(lines_101), (
            "M=101 should contain UniRef header with metadata (10648|...)"
        )
        assert "MATCH2\n" in lines_101, "M=101 should contain match sequence after metadata header"
        # Should NOT contain M=102 content
        assert ">102\n" not in lines_101, "M=101 should not contain >102"
        assert "SEQUENCE2\n" not in lines_101, "M=101 should not contain SEQUENCE2"

        # Check M=102 section - should contain the other UniRef header with metadata
        lines_102 = result[102]
        assert ">102\n" in lines_102, "M=102 should start with >102"
        assert "SEQUENCE2\n" in lines_102, "M=102 should contain SEQUENCE2"
        assert ">UniRef100_B0B456\n" in lines_102, "M=102 should contain simple UniRef header"
        assert ">10566|scaffold|+1|5" in "".join(lines_102), (
            "M=102 should contain UniRef header with metadata (10566|...)"
        )
        assert "MATCH4\n" in lines_102, "M=102 should contain match sequence after metadata header"
        # Should NOT contain M=101 content
        assert ">101\n" not in lines_102, "M=102 should not contain >101"
        assert "SEQUENCE1\n" not in lines_102, "M=102 should not contain SEQUENCE1"
        assert ">10648|Ga0318525_12252441_1|+3|11" not in "".join(lines_102), (
            "M=102 should not contain metadata header from M=101"
        )
