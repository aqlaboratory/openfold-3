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
"""Tests for DatasetCache LMDB construction and env lifecycle."""

import lmdb
import pytest

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.caches.lmdb import LMDBDict


class TestDatasetCacheFromLMDB:
    def test_from_lmdb_sets_lmdb_env(self, lmdb_cache):
        """read_datacache(lmdb_dir) should set _lmdb_env to a live lmdb.Environment."""
        assert lmdb_cache._lmdb_env is not None
        assert isinstance(lmdb_cache._lmdb_env, lmdb.Environment)

    def test_from_json_lmdb_env_is_none(self, json_cache):
        """from_json should leave _lmdb_env as None."""
        cache = read_datacache(json_cache)
        assert cache._lmdb_env is None

    @pytest.mark.parametrize(
        "field",
        ["structure_data", "reference_molecule_data"],
        ids=["structure_data", "reference_molecule_data"],
    )
    def test_from_lmdb_fields_are_lmdb_dicts(self, lmdb_cache, field):
        """LMDB-backed caches should use LMDBDict, not plain dicts."""
        assert isinstance(getattr(lmdb_cache, field), LMDBDict)

    @pytest.mark.parametrize(
        ("field", "key", "attr", "expected"),
        [
            (
                "structure_data",
                "test0",
                "chains.0.alignment_representative_id",
                "test_id0",
            ),
            (
                "structure_data",
                "test1",
                "chains.0.alignment_representative_id",
                "test_id1",
            ),
            ("reference_molecule_data", "ALA", "canonical_smiles", "C[C@H](N)C(=O)O"),
        ],
        ids=[
            "structure_data-test0",
            "structure_data-test1",
            "reference_molecule_data-ALA",
        ],
    )
    def test_from_lmdb_lazy_read(self, lmdb_cache, field, key, attr, expected):
        """Individual key lookups should return correct data through the live env."""
        value = getattr(lmdb_cache, field)[key]
        for part in attr.split("."):
            value = value[part] if isinstance(value, dict) else getattr(value, part)
        assert value == expected

    def test_from_lmdb_env_shared_across_dicts(self, lmdb_cache):
        """Both LMDBDicts must share the same env (LMDB forbids multiple opens)."""
        assert (
            lmdb_cache.structure_data._lmdb_env
            is lmdb_cache.reference_molecule_data._lmdb_env
        )

    @pytest.mark.parametrize(
        "field",
        ["structure_data", "reference_molecule_data"],
        ids=["structure_data", "reference_molecule_data"],
    )
    def test_from_lmdb_missing_key_raises(self, lmdb_cache, field):
        """Accessing a non-existent key should raise KeyError."""
        with pytest.raises(KeyError):
            getattr(lmdb_cache, field)["nonexistent"]
