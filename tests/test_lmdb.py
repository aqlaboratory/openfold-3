"""Helper functions to test the lmdb dict"""

import json
import lmdb
from openfold3.core.data.io.dataset_cache import read_datacache
import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.data.primitives.caches.lmdb import convert_datacache_to_lmdb
from openfold3.core.data.primitives.caches.lmdb import LMDBDict

TEST_DATASET_CONFIG = {
    "_type": "ProteinMonomerDatasetCache",
    "name": "DummySet",
    "structure_data": {
        "test0": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id0",
                    "template_ids": [],
                },
            },
        },
        "test1": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id1",
                    "template_ids": [],
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


class TestLMDBDict:
    def test_lmdb_roundtrip(self, tmp_path):
        # Save dummy json
        test_config_json = tmp_path / "test_config.json"
        with open(test_config_json, "w") as f:
            json.dump(TEST_DATASET_CONFIG, f, indent=4)

        # Create LMDB
        test_lmdb_dir = tmp_path / "test_lmdb"
        map_size = 20 * 1024
        convert_datacache_to_lmdb(test_config_json, test_lmdb_dir, map_size)

        # read lmdb
        lmdb_cache = read_datacache(test_lmdb_dir)
        # compare with json reloaded cache
        expected_cache = read_datacache(test_config_json)

        assert lmdb_cache == expected_cache
