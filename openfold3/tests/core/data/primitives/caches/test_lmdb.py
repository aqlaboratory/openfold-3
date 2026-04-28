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

"""Tests for the LMDB dict and convert_datacache_to_lmdb."""

import json

import lmdb
import pytest
from conftest import TEST_DATASET_CONFIG

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.caches.lmdb import convert_datacache_to_lmdb


class TestLMDBDict:
    def test_lmdb_roundtrip(self, json_cache, lmdb_cache):
        expected_cache = read_datacache(json_cache)
        assert lmdb_cache == expected_cache


class TestConvertDatacacheToLMDB:
    def test_env_closed_after_write(self, tmp_path, json_cache):
        """The write-path context manager should close the env on return.

        LMDB forbids opening the same directory twice in one process. If
        convert_datacache_to_lmdb leaked the env, this second open would raise
        lmdb.Error.
        """
        lmdb_dir = tmp_path / "lmdb"
        convert_datacache_to_lmdb(json_cache, lmdb_dir, map_size=2 * (1024**2))

        # Would raise "already open in this process" if the write env leaked
        try:
            env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True)
            env.close()
        except lmdb.Error as exc:
            pytest.fail(f"lmdb env was not closed after write — re-open raised: {exc}")

    @pytest.mark.parametrize(
        ("prefix", "expected_keys"),
        [
            ("structure_data", {"structure_data:test0", "structure_data:test1"}),
            ("reference_molecule_data", {"reference_molecule_data:ALA"}),
        ],
        ids=["structure_data", "reference_molecule_data"],
    )
    def test_written_keys(self, tmp_path, json_cache, prefix, expected_keys):
        """All entries should be written as prefixed keys."""
        lmdb_dir = tmp_path / "lmdb"
        convert_datacache_to_lmdb(json_cache, lmdb_dir, map_size=2 * (1024**2))

        with (
            lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True) as env,
            env.begin() as txn,
            txn.cursor() as cursor,
        ):
            keys = {k.decode() for k, _ in cursor if k.decode().startswith(prefix)}
        assert keys == expected_keys

    def test_metadata_keys_written(self, tmp_path, json_cache):
        """_type and name metadata should be stored."""
        lmdb_dir = tmp_path / "lmdb"
        convert_datacache_to_lmdb(json_cache, lmdb_dir, map_size=2 * (1024**2))

        with (
            lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True) as env,
            env.begin() as txn,
        ):
            _type = json.loads(txn.get(b"_type").decode())
            name = json.loads(txn.get(b"name").decode())

        assert _type == TEST_DATASET_CONFIG["_type"]
        assert name == TEST_DATASET_CONFIG["name"]
