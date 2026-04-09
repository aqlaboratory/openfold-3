"""Tests for the is_dir (LMDB) branch of read_datacache."""

import pytest

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.caches.lmdb import LMDBDict


class TestReadDatacacheLMDB:
    def test_type_peek_env_cleaned_up(self, lmdb_dir):
        """read_datacache opens a short-lived env to peek at _type, then closes it.

        If the peek env leaked, from_lmdb's lmdb.open on the same directory
        would raise "already open in this process".  A successful return
        proves the peek env was closed by the context manager.
        """
        cache = read_datacache(lmdb_dir)
        assert cache._lmdb_env is not None

    def test_returns_correct_type(self, lmdb_dir):
        """Should infer the correct DatasetCache subclass from _type."""
        cache = read_datacache(lmdb_dir)
        assert type(cache).__name__ == "ProteinMonomerDatasetCache"

    @pytest.mark.parametrize(
        "field",
        ["structure_data", "reference_molecule_data"],
        ids=["structure_data", "reference_molecule_data"],
    )
    def test_fields_are_lmdb_dicts(self, lmdb_dir, field):
        """LMDB-backed fields should be LMDBDict instances, not plain dicts."""
        cache = read_datacache(lmdb_dir)
        assert isinstance(getattr(cache, field), LMDBDict)

    def test_invalid_path_raises(self, tmp_path):
        """A path that is neither file nor directory should raise ValueError."""
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="Invalid datacache path"):
            read_datacache(bogus)

    def test_lmdb_env_is_readonly(self, lmdb_dir):
        """The env held by from_lmdb should be opened readonly."""
        cache = read_datacache(lmdb_dir)
        env_flags = cache._lmdb_env.flags()
        assert env_flags["readonly"] is True
