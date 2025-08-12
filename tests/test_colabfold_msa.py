"""Tests to check handling of colabofold MSA data."""

import json
import textwrap
from unittest.mock import patch

import pandas as pd
import pytest

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
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
        # dummy tsv output
        raw_main_dir = tmp_path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {0: [101, 101, 102], 1: ["test_A", "test_B", "test_C"], 2: [0, 1, 2]}
        ).to_csv(raw_main_dir / "pdb70.m8", header=False, index=False, sep="\t")

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
        raw_main_dir = tmp_path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {0: [101, 101, 102], 1: ["test_A", "test_B", "test_C"], 2: [0, 1, 2]}
        ).to_csv(raw_main_dir / "pdb70.m8", header=False, index=False, sep="\t")

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

        # dummy tsv output
        raw_main_dir = tmp_path / "raw" / "main"
        raw_main_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {0: [101, 101, 102], 1: ["test_A", "test_B", "test_C"], 2: [0, 1, 2]}
        ).to_csv(raw_main_dir / "pdb70.m8", header=False, index=False, sep="\t")

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
            )
            inference_spec = InferenceDatasetSpec(config=inference_config)

            data_config = DataModuleConfig(
                datasets=[inference_spec],
                batch_size=1,
                epoch_len=1,
                num_epochs=1,
            )

            data_module = DataModule(data_config)

            print("making prediction for sequence:", sequence)
            data_module.setup()
            dataloader = data_module.predict_dataloader()

            expected_msa = 4  # based on _construct_dummy_a3m
            expected_shape = (1, expected_msa, len(sequence), 32)
            for batch in dataloader:
                assert batch["msa"].shape == expected_shape

        # Test contents of mapping file after all runs
        with open(tmp_path / "mappings/seq_to_rep_id.json") as f:
            assert set(json.load(f).keys()) == set(test_sequences), (
                "Expected all test sequences to be present in the mapping file"
            )
