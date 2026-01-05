"""Smoke tests for preparse_alignments_of3.py script."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scripts.data_preprocessing.preparse_alignments_of3 import main

# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data" / "alignments"


class TestPreparseAlignmentsOf3:
    """Smoke tests for preparse_alignments_of3.py script."""

    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_preparse_databases(self, cli_runner, tmp_path):
        """Test preparsing alignments with a single database (uniref90_hits)."""
        max_seq_counts = json.dumps({"uniref90_hits": 100, "uniprot_hits": 50})

        result = cli_runner.invoke(
            main,
            [
                "--alignments_directory",
                str(TEST_DATA_DIR),
                "--alignment_array_directory",
                str(tmp_path),
                "--max_seq_counts",
                max_seq_counts,
                "--num_workers",
                "1",
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.output}"

        # Check that npz files were created for both chains
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 6, (
            f"Expected 6 npz files, got {len(npz_files)}: {[f.name for f in npz_files]}"
        )

        chain_names = {f.stem for f in npz_files}
        assert "2q2k_A" in chain_names
        assert "2q2k_B" in chain_names

        # Check contents of one npz file
        npz_data = np.load(tmp_path / "2q2k_A.npz", allow_pickle=True)
        array_names = list(npz_data.keys())

        # Should have uniref90_hits with msa, deletion_matrix, metadata
        assert "uniref90_hits" in array_names
        assert "uniprot_hits" in array_names

    def test_invalid_max_seq_counts_json(self, cli_runner, tmp_path):
        """Test that invalid JSON raises an error."""
        result = cli_runner.invoke(
            main,
            [
                "--alignments_directory",
                str(TEST_DATA_DIR),
                "--alignment_array_directory",
                str(tmp_path),
                "--max_seq_counts",
                "not valid json",
                "--num_workers",
                "1",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid max_seq_counts JSON string" in result.output

    def test_invalid_field_in_max_seq_counts(self, cli_runner, tmp_path):
        """Test that unknown fields in max_seq_counts raise an error."""
        max_seq_counts = json.dumps({"unknown_database": 100})

        result = cli_runner.invoke(
            main,
            [
                "--alignments_directory",
                str(TEST_DATA_DIR),
                "--alignment_array_directory",
                str(tmp_path),
                "--max_seq_counts",
                max_seq_counts,
                "--num_workers",
                "1",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid max_seq_counts JSON string" in result.output
