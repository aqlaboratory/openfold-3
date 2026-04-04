"""Tests for ``openfold3.core.data.tools.utils``."""

import getpass
import tempfile
from pathlib import Path
from unittest.mock import patch

from openfold3.core.data.tools.utils import get_of3_tmpdir


class TestGetOf3Tmpdir:
    """Tests for get_of3_tmpdir."""

    def test_returns_path(self):
        result = get_of3_tmpdir("test_subdir")
        assert isinstance(result, Path)

    def test_directory_is_created(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            result = get_of3_tmpdir("mysubdir")

        assert result.is_dir()

    def test_contains_username(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            result = get_of3_tmpdir("subdir")

        assert f"of3-of-{getpass.getuser()}" in str(result)

    def test_subdir_appended(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            result = get_of3_tmpdir("colabfold_msas")

        assert result.name == "colabfold_msas"
        assert result.parent.name == f"of3-of-{getpass.getuser()}"

    def test_no_subdir(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            result = get_of3_tmpdir()

        assert result.name == f"of3-of-{getpass.getuser()}"
        assert result.is_dir()

    def test_respects_tmpdir_env(self, tmp_path, monkeypatch):
        custom_tmp = tmp_path / "custom_tmp"
        custom_tmp.mkdir()
        monkeypatch.setenv("TMPDIR", str(custom_tmp))
        # Force tempfile to re-evaluate TMPDIR
        tempfile.tempdir = None

        result = get_of3_tmpdir("subdir")

        assert str(result).startswith(str(custom_tmp))

    def test_idempotent(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            first = get_of3_tmpdir("subdir")
            second = get_of3_tmpdir("subdir")

        assert first == second

    def test_different_subdirs_are_isolated(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            a = get_of3_tmpdir("aaa")
            b = get_of3_tmpdir("bbb")

        assert a != b
        assert a.parent == b.parent

    def test_different_users_are_isolated(self, tmp_path):
        with patch.object(tempfile, "gettempdir", return_value=str(tmp_path)):
            real = get_of3_tmpdir("data")
            with patch.object(getpass, "getuser", return_value="other_user"):
                other = get_of3_tmpdir("data")

        assert real != other
        assert "of3-of-other_user" in str(other)
