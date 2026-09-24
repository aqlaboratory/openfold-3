"""Tests for the RCSB GraphQL API helpers in ``openfold3.core.data.tools.rscb``.

Tests marked ``@pytest.mark.vcr`` use *pytest-recording* (vcrpy) to replay
HTTP responses from YAML cassettes stored alongside this file in
``cassettes/``.

Generating cassettes for the first time::

    pytest openfold3/tests/core/data/tools/test_rscb.py --vcr-record=all

Re-recording after RCSB schema changes or new test methods::

    pytest openfold3/tests/core/data/tools/test_rscb.py --vcr-record=new_episodes

In CI the cassettes are replayed without network access (the default
``--vcr-record=none`` mode).
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import requests

from openfold3.core.data.tools import rscb
from openfold3.core.data.tools.rscb import (
    fetch_label_to_author_chain_ids,
    get_model_ranking_fit,
)


class TestFetchLabelToAuthorChainIds:
    """Tests for fetch_label_to_author_chain_ids (recorded RCSB responses)."""

    @pytest.mark.vcr
    def test_1rnb_label_to_author(self):
        """1RNB: label chain B -> author chain A (protein)."""
        result = fetch_label_to_author_chain_ids({"1rnb"})

        assert "1rnb" in result
        l2a = result["1rnb"]
        assert l2a["B"] == "A"
        assert l2a["A"] == "C"

    @pytest.mark.vcr
    def test_identity_mapping(self):
        """4PQX: label chain IDs match author chain IDs."""
        result = fetch_label_to_author_chain_ids({"4pqx"})

        assert "4pqx" in result
        assert result["4pqx"]["A"] == "A"

    @pytest.mark.vcr
    def test_batch_query(self):
        """Multiple PDB IDs are fetched in a single request."""
        result = fetch_label_to_author_chain_ids({"1rnb", "4pqx"})

        assert "1rnb" in result
        assert "4pqx" in result

    def test_empty_set(self):
        """Empty input returns empty dict without API call."""
        assert fetch_label_to_author_chain_ids(set()) == {}


def _graphql_response(pdb_ids: list[str]) -> MagicMock:
    """A successful RCSB response with an identity chain mapping per entry."""
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


def _http_error(status_code: int) -> requests.HTTPError:
    return requests.HTTPError(response=MagicMock(status_code=status_code))


@patch.object(rscb.time, "sleep")
@patch.object(rscb.requests, "post")
class TestFetchLabelToAuthorChainIdsRobustness:
    """Retry and batching behaviour, with the network mocked out."""

    def test_uses_long_timeout(self, mock_post, _sleep):
        mock_post.return_value = _graphql_response(["1abc"])

        fetch_label_to_author_chain_ids({"1abc"})

        assert mock_post.call_args.kwargs["timeout"] == rscb._RCSB_TIMEOUT_S
        assert rscb._RCSB_TIMEOUT_S > 30

    def test_retries_after_read_timeout(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            requests.ReadTimeout("read timed out"),
            _graphql_response(["1abc"]),
        ]

        result = fetch_label_to_author_chain_ids({"1abc"})

        assert result == {"1abc": {"A": "A"}}
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(rscb._RCSB_BACKOFF_S)

    def test_retries_on_throttling_and_5xx(self, mock_post, _sleep):
        mock_post.side_effect = [
            _http_error(503),
            _http_error(429),
            _graphql_response(["1abc"]),
        ]

        assert fetch_label_to_author_chain_ids({"1abc"}) == {"1abc": {"A": "A"}}
        assert mock_post.call_count == 3

    def test_gives_up_after_max_attempts(self, mock_post, _sleep):
        mock_post.side_effect = requests.ReadTimeout("read timed out")

        with pytest.raises(RuntimeError, match="Failed to fetch chain ID mappings"):
            fetch_label_to_author_chain_ids({"1abc"})

        assert mock_post.call_count == rscb._RCSB_MAX_ATTEMPTS

    def test_client_error_is_not_retried(self, mock_post, _sleep):
        mock_post.side_effect = _http_error(400)

        with pytest.raises(RuntimeError, match="Failed to fetch chain ID mappings"):
            fetch_label_to_author_chain_ids({"1abc"})

        assert mock_post.call_count == 1

    def test_large_id_sets_are_batched(self, mock_post, _sleep):
        """A CI-sized ID set goes out as several bounded requests."""
        LARGE_NUM_IDS = 218
        pdb_ids = {f"{i:04d}" for i in range(LARGE_NUM_IDS)}
        mock_post.side_effect = lambda _url, json, **_: _graphql_response(
            json["variables"]["ids"]
        )

        result = fetch_label_to_author_chain_ids(pdb_ids)

        batch_size = rscb._RCSB_CHAIN_MAPPING_BATCH_SIZE
        sent = [
            call.kwargs["json"]["variables"]["ids"] for call in mock_post.call_args_list
        ]
        assert len(sent) == math.ceil(LARGE_NUM_IDS / batch_size)
        assert all(len(batch) <= batch_size for batch in sent)
        assert sorted(pid for batch in sent for pid in batch) == sorted(pdb_ids)
        assert set(result) == pdb_ids


class TestGetModelRankingFit:
    """Tests for get_model_ranking_fit (recorded RCSB responses)."""

    @pytest.mark.vcr
    def test_entry_with_ligands(self):
        """4PQX has ligands with ranking_model_fit scores."""
        result = get_model_ranking_fit("4pqx")

        assert isinstance(result, dict)
        assert len(result) > 0
        for rcsb_id, score in result.items():
            assert rcsb_id.startswith("4PQX.")
            assert isinstance(score, (int, float))

    @pytest.mark.vcr
    def test_entry_without_ligands(self):
        """1RNB (protein-only) returns empty dict."""
        result = get_model_ranking_fit("1rnb")

        assert result == {}

    @pytest.mark.vcr
    def test_nonexistent_entry(self):
        """Invalid PDB ID returns empty dict without raising."""
        result = get_model_ranking_fit("0000")

        assert result == {}
