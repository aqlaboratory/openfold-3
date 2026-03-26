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

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from download_of3_databases import (
    S3_PREFIX,
    download,
    download_from_s3,
    format_size,
    list_databases,
    parse_args,
)

# --- Fixtures and helpers ---


def make_download_args(
    output_dir,
    *,
    download_bfd=False,
    download_cfdb=False,
    download_rna_dbs=False,
    jackhmmer_dbs=None,
    hhblits_dbs=None,
):
    """Create a Namespace with download args, using sensible defaults."""
    return Namespace(
        output_dir=output_dir,
        download_bfd=download_bfd,
        download_cfdb=download_cfdb,
        download_rna_dbs=download_rna_dbs,
        jackhmmer_dbs=jackhmmer_dbs if jackhmmer_dbs is not None else [],
        hhblits_dbs=hhblits_dbs,
    )


# --- Helper function tests ---


@pytest.mark.parametrize(
    "size_bytes,expected",
    [
        (500, "500.0 B"),
        (1024, "1.0 KB"),
        (2048, "2.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024 * 1024 * 1024, "1.0 GB"),
        (1024 * 1024 * 1024 * 1024, "1.0 TB"),
    ],
)
def test_format_size(size_bytes, expected):
    assert format_size(size_bytes) == expected


# --- parse_args tests ---


def test_parse_args_list_command():
    with patch("sys.argv", ["script", "list"]):
        args = parse_args()
    assert args.command == "list"


def test_parse_args_download_defaults():
    with patch("sys.argv", ["script", "download"]):
        args = parse_args()
    assert args.command == "download"
    assert args.output_dir == "./alignment_dbs"
    assert args.download_bfd is False
    assert args.download_cfdb is False
    assert args.download_rna_dbs is False
    assert args.jackhmmer_dbs is None
    assert args.hhblits_dbs is None


def test_parse_args_download_custom_output_dir():
    with patch("sys.argv", ["script", "download", "--output-dir", "/custom/path"]):
        args = parse_args()
    assert args.output_dir == "/custom/path"


@pytest.mark.parametrize(
    ["flag", "attr"],
    [
        ("--download-bfd", "download_bfd"),
        ("--download-cfdb", "download_cfdb"),
        ("--download-rna-dbs", "download_rna_dbs"),
    ],
)
def test_parse_args_download_flag(flag, attr):
    with patch("sys.argv", ["script", "download", flag]):
        args = parse_args()
    assert getattr(args, attr) is True


def test_parse_args_custom_jackhmmer_dbs():
    with patch(
        "sys.argv", ["script", "download", "--jackhmmer-dbs", "uniref90", "pdb_seqres"]
    ):
        args = parse_args()
    assert args.jackhmmer_dbs == ["uniref90", "pdb_seqres"]


def test_parse_args_custom_hhblits_dbs():
    with patch("sys.argv", ["script", "download", "--hhblits-dbs", "uniref30", "bfd"]):
        args = parse_args()
    assert args.hhblits_dbs == ["uniref30", "bfd"]


# --- download_from_s3 tests ---


def test_download_from_s3_command():
    with patch("download_of3_databases.sp.run") as mock_run:
        download_from_s3(
            bucket="test-bucket", key="path/file.gz", destination="/tmp/file.gz"
        )
        mock_run.assert_called_once_with(
            [
                "aws",
                "s3",
                "cp",
                "--no-sign-request",
                "s3://test-bucket/path/file.gz",
                "/tmp/file.gz",
            ],
            check=True,
        )


# --- list_databases tests ---


def test_list_databases_parses_s3_json_output(capsys):
    mock_json = {
        "Contents": [
            {"Key": "alignment_databases/test.fasta.gz", "Size": 1234567890},
            {"Key": "alignment_databases/uniref90.fasta.gz", "Size": 9876543210},
        ]
    }

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_json)

    with patch("download_of3_databases.sp.run", return_value=mock_result):
        list_databases()

    captured = capsys.readouterr()
    assert "test.fasta.gz" in captured.out
    assert "uniref90.fasta.gz" in captured.out
    assert "✓" in captured.out  # uniref90 is a known database
    assert "Protein" in captured.out  # uniref90 is a protein database


def test_list_databases_shows_human_readable_sizes(capsys):
    mock_json = {
        "Contents": [
            {"Key": "alignment_databases/test.tar.gz", "Size": 1073741824},  # 1 GB
        ]
    }

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_json)

    with patch("download_of3_databases.sp.run", return_value=mock_result):
        list_databases()

    captured = capsys.readouterr()
    assert "1.0 GB" in captured.out


def test_list_databases_shows_type_column(capsys):
    mock_json = {
        "Contents": [
            {"Key": "alignment_databases/uniref90.fasta.gz", "Size": 1000},
            {"Key": "alignment_databases/rfam.fasta.gz", "Size": 2000},
        ]
    }

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_json)

    with patch("download_of3_databases.sp.run", return_value=mock_result):
        list_databases()

    captured = capsys.readouterr()
    assert "Protein" in captured.out  # uniref90 is protein
    assert "DNA/RNA" in captured.out  # rfam is DNA/RNA


# --- download jackhmmer tests ---


def test_download_skips_existing_jackhmmer_databases():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
    ):
        # Create existing unzipped file
        db_dir = Path(tmpdir) / "uniref90"
        db_dir.mkdir()
        (db_dir / "uniref90.fasta").touch()

        args = make_download_args(tmpdir, jackhmmer_dbs=["uniref90"], hhblits_dbs=[])
        download(args)

        # Should not download uniref90 since it exists
        mock_download.assert_not_called()


def test_download_downloads_custom_jackhmmer_dbs():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
    ):
        args = make_download_args(
            tmpdir, jackhmmer_dbs=["uniref90", "pdb_seqres"], hhblits_dbs=[]
        )
        download(args)

        calls = mock_download.call_args_list
        downloaded_keys = [c.kwargs["key"] for c in calls]
        assert f"{S3_PREFIX}/uniref90.fasta.gz" in downloaded_keys
        assert f"{S3_PREFIX}/pdb_seqres.fasta.gz" in downloaded_keys


def test_download_includes_rna_dbs_when_flag_set():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
    ):
        args = make_download_args(
            tmpdir, download_rna_dbs=True, jackhmmer_dbs=["uniref90"], hhblits_dbs=[]
        )
        download(args)

        calls = mock_download.call_args_list
        downloaded_keys = [c.kwargs["key"] for c in calls]
        assert f"{S3_PREFIX}/rfam.fasta.gz" in downloaded_keys
        assert f"{S3_PREFIX}/rnacentral.fasta.gz" in downloaded_keys


# --- download hhblits tests ---


def test_download_skips_existing_hhblits_databases():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
    ):
        # Create existing directory (signals database exists)
        db_dir = Path(tmpdir) / "uniref30"
        db_dir.mkdir()

        args = make_download_args(tmpdir)
        download(args)

        # uniref30 should be skipped
        for call_args in mock_download.call_args_list:
            assert "uniref30" not in call_args.kwargs.get("key", "")


@pytest.mark.parametrize(
    "flag_name,db_name",
    [
        ("download_bfd", "bfd"),
        ("download_cfdb", "cfdb"),
    ],
)
def test_download_optional_db_when_flag_set(flag_name, db_name):
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
        patch.object(Path, "unlink"),
    ):
        args = make_download_args(
            tmpdir,
            download_bfd=(flag_name == "download_bfd"),
            download_cfdb=(flag_name == "download_cfdb"),
        )
        download(args)

        calls = mock_download.call_args_list
        downloaded_keys = [c.kwargs["key"] for c in calls]
        assert f"{S3_PREFIX}/{db_name}.tar.gz" in downloaded_keys


def test_download_custom_hhblits_dbs_ignores_bfd_cfdb_flags():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run"),
        patch("download_of3_databases.download_from_s3") as mock_download,
        patch.object(Path, "unlink"),
    ):
        # bfd/cfdb flags should be ignored when custom hhblits_dbs is set
        args = make_download_args(
            tmpdir, download_bfd=True, download_cfdb=True, hhblits_dbs=["custom_db"]
        )
        download(args)

        calls = mock_download.call_args_list
        downloaded_keys = [c.kwargs["key"] for c in calls]
        # Only custom_db should be downloaded, not bfd or cfdb
        assert len(downloaded_keys) == 1
        assert f"{S3_PREFIX}/custom_db.tar.gz" in downloaded_keys


# --- unzip/extract tests ---


def test_download_gunzip_called_for_jackhmmer_dbs():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run") as mock_run,
        patch("download_of3_databases.download_from_s3"),
    ):
        args = make_download_args(tmpdir, jackhmmer_dbs=["uniref90"], hhblits_dbs=[])
        download(args)

        gunzip_calls = [c for c in mock_run.call_args_list if "gunzip" in c[0][0]]
        assert len(gunzip_calls) == 1


def test_download_tar_called_for_hhblits_dbs():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("download_of3_databases.sp.run") as mock_run,
        patch("download_of3_databases.download_from_s3"),
        patch.object(Path, "unlink"),
    ):
        args = make_download_args(tmpdir, hhblits_dbs=["testdb"])
        download(args)

        tar_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "tar"]
        assert len(tar_calls) == 1
        assert "xzf" in tar_calls[0][0][0]
