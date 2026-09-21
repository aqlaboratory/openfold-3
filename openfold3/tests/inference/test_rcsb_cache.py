"""Checks the session-wide RCSB memoisation applied by ``conftest.py``.

Not marked ``slow``: no GPU and no network, the RCSB transport is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

import openfold3.core.data.tools.colabfold_msa_server as colabfold_msa_server
from openfold3.core.data.tools import rscb


def _response(pdb_ids: list[str]) -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": {
            "entries": [
                {
                    "rcsb_id": pid.upper(),
                    "polymer_entities": [
                        {
                            "rcsb_polymer_entity_container_identifiers": {
                                "asym_ids": ["A"],
                                "auth_asym_ids": ["A"],
                            }
                        }
                    ],
                }
                for pid in pdb_ids
            ]
        }
    }
    return resp


@patch.object(rscb.requests, "post")
def test_same_id_set_is_fetched_once_per_session(
    mock_post, cache_rcsb_chain_id_mappings
):
    mock_post.side_effect = lambda _url, json, **_: _response(json["variables"]["ids"])
    cache_rcsb_chain_id_mappings.clear()
    fetch = colabfold_msa_server.fetch_label_to_author_chain_ids

    first = fetch({"1ubq", "1aar"})
    second = fetch({"1aar", "1ubq"})
    other = fetch({"1ubq"})

    assert first == second == {"1ubq": {"A": "A"}, "1aar": {"A": "A"}}
    assert other == {"1ubq": {"A": "A"}}
    # One request per distinct ID set: the repeat was served from the cache.
    assert mock_post.call_count == 2
    assert set(cache_rcsb_chain_id_mappings) == {
        frozenset({"1ubq", "1aar"}),
        frozenset({"1ubq"}),
    }


@patch.object(rscb.requests, "post")
@patch.object(rscb.time, "sleep")
def test_failed_fetch_is_not_cached(_sleep, mock_post, cache_rcsb_chain_id_mappings):
    cache_rcsb_chain_id_mappings.clear()
    fetch = colabfold_msa_server.fetch_label_to_author_chain_ids
    mock_post.side_effect = requests.ReadTimeout("read timed out")

    with pytest.raises(RuntimeError):
        fetch({"1ubq"})
    assert not cache_rcsb_chain_id_mappings

    mock_post.side_effect = lambda _url, json, **_: _response(json["variables"]["ids"])
    assert fetch({"1ubq"}) == {"1ubq": {"A": "A"}}
