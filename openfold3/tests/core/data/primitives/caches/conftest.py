"""Shared fixtures for cache tests."""

import json

import pytest

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.caches.lmdb import convert_datacache_to_lmdb

TEST_DATASET_CONFIG = {
    "_type": "ProteinMonomerDatasetCache",
    "name": "DummySet",
    "structure_data": {
        "test0": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id0",
                    "template_ids": [],
                    "index": 0,
                },
            },
        },
        "test1": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id1",
                    "template_ids": [],
                    "index": 1,
                },
            },
        },
    },
    "reference_molecule_data": {
        "ALA": {
            "conformer_gen_strategy": "default",
            "fallback_conformer_pdb_id": None,
            "canonical_smiles": "C[C@H](N)C(=O)O",
            "set_fallback_to_nan": False,
        },
    },
}


@pytest.fixture()
def json_cache(tmp_path):
    """Write the test config to a JSON file and return the path."""
    path = tmp_path / "cache.json"
    with open(path, "w") as f:
        json.dump(TEST_DATASET_CONFIG, f, indent=4)
    return path


@pytest.fixture()
def lmdb_cache(tmp_path, json_cache):
    """Convert the JSON cache to LMDB and return a DatasetCache loaded from it."""
    lmdb_dir = tmp_path / "cache_lmdb"
    convert_datacache_to_lmdb(json_cache, lmdb_dir, map_size=2 * (1024**2))
    return read_datacache(lmdb_dir)


@pytest.fixture()
def lmdb_dir(tmp_path, json_cache):
    """Convert the JSON cache to LMDB, return the LMDB directory path."""
    from openfold3.core.data.primitives.caches.lmdb import convert_datacache_to_lmdb

    lmdb_path = tmp_path / "cache_lmdb_dir"
    convert_datacache_to_lmdb(json_cache, lmdb_path, map_size=2 * (1024**2))
    return lmdb_path
