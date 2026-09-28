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

import logging
import time
from collections.abc import Iterable

import requests

logger = logging.getLogger(__name__)

_RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

# Per-request timeout in seconds.
_RCSB_TIMEOUT_S = 120

# Attempts per request before giving up, and the base of the exponential backoff.
_RCSB_MAX_ATTEMPTS = 3
_RCSB_BACKOFF_S = 3.0

# Maximum number of entry IDs per chain-mapping request. Smaller batches keep any
# single request well inside the timeout even when the server is slow.
_RCSB_CHAIN_MAPPING_BATCH_SIZE = 50

# HTTP statuses worth a retry: throttling and transient server-side failures.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_CHAIN_MAPPING_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        asym_ids
        auth_asym_ids
      }
    }
  }
}
"""


def _post_graphql_with_retry(payload: dict) -> requests.Response:
    """POST a GraphQL payload to RCSB, retrying transient failures with backoff.

    Retries on timeouts, connection errors and throttling/5xx responses. Any other
    HTTP error (e.g. a malformed query) is raised on the first attempt.

    Args:
        payload: JSON body of the request (``query`` and ``variables``).

    Returns:
        The successful response.

    Raises:
        requests.RequestException: If every attempt fails.
    """
    last_error: Exception | None = None
    for attempt in range(1, _RCSB_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                _RCSB_GRAPHQL_URL, json=payload, timeout=_RCSB_TIMEOUT_S
            )
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
        except requests.HTTPError as e:
            if (
                e.response is None
                or e.response.status_code not in _RETRYABLE_STATUS_CODES
            ):
                raise
            last_error = e
        if attempt < _RCSB_MAX_ATTEMPTS:
            delay = _RCSB_BACKOFF_S * 2 ** (attempt - 1)
            logger.warning(
                "RCSB request failed (attempt %d/%d): %s. Retrying in %.0fs.",
                attempt,
                _RCSB_MAX_ATTEMPTS,
                last_error,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_label_to_author_chain_ids(
    pdb_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Fetch label-to-author chain ID mappings from the RCSB PDB GraphQL API.

    Sends the PDB IDs in batches of at most ``_RCSB_CHAIN_MAPPING_BATCH_SIZE`` (each
    batch retried on transient failures) and returns a nested dict mapping
    ``entry_id`` → ``label_asym_id`` → ``author_chain_id``.

    Args:
        pdb_ids: Set of PDB entry IDs (e.g. ``{"4pqx", "1rnb"}``).

    Returns:
        Nested dict: ``entry_id`` (lower-case) → ``label_asym_id`` →
        ``author_chain_id``.

    Raises:
        RuntimeError: If the RCSB API request fails.
    """
    if not pdb_ids:
        return {}

    result: dict[str, dict[str, str]] = {}
    for batch in _batched(sorted(pdb_ids), _RCSB_CHAIN_MAPPING_BATCH_SIZE):
        try:
            resp = _post_graphql_with_retry(
                {"query": _CHAIN_MAPPING_QUERY, "variables": {"ids": batch}}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch chain ID mappings from RCSB for "
                f"{len(pdb_ids)} entries. Cannot proceed without chain ID "
                f"re-mapping."
            ) from e

        data = resp.json().get("data", {})
        entries = data.get("entries") or []

        for entry in entries:
            entry_id = entry["rcsb_id"].lower()
            label_to_author: dict[str, str] = {}
            for entity in entry.get("polymer_entities") or []:
                ids = entity["rcsb_polymer_entity_container_identifiers"]
                for asym_id, auth_id in zip(
                    ids["asym_ids"], ids["auth_asym_ids"], strict=True
                ):
                    label_to_author[asym_id] = auth_id
            result[entry_id] = label_to_author

    return result


_MODEL_RANKING_FIT_QUERY = """
query GetRankingFit($pdb_id: String!) {
    entry(entry_id: $pdb_id) {
        nonpolymer_entities {
            nonpolymer_entity_instances {
                rcsb_id
                rcsb_nonpolymer_instance_validation_score {
                    ranking_model_fit
                }
            }
        }
    }
}
"""


# TODO: Do this in preprocessing instead to avoid it going out-of-sync with the data?
def get_model_ranking_fit(pdb_id: str) -> dict[str, float]:
    """Fetch model ranking fit entries for all ligands of a single PDB entry.

    Uses the RCSB PDB GraphQL API to fetch the model ranking fit values for
    all ligands in a single PDB entry. Note that this function will always
    fetch from the newest version of the PDB and can therefore occasionally
    give incorrect results for old datasets whose structures have been updated
    since.

    Args:
        pdb_id: PDB entry ID (e.g. ``"4pqx"``).

    Returns:
        Dictionary mapping ``rcsb_id`` (e.g. ``"4PQX.C"``) to its
        ``ranking_model_fit`` score.  Returns an empty dict on failure.
    """
    response = requests.post(
        _RCSB_GRAPHQL_URL,
        json={"query": _MODEL_RANKING_FIT_QUERY, "variables": {"pdb_id": pdb_id}},
        timeout=_RCSB_TIMEOUT_S,
    )

    if response.status_code != 200:
        logger.warning("RCSB request failed with status code %d", response.status_code)
        return {}

    try:
        data = response.json()
        entry_data = data.get("data", {}).get("entry", {})
        if not entry_data:
            return {}

        extracted_data: dict[str, float] = {}
        for entity in entry_data.get("nonpolymer_entities") or []:
            for instance in entity.get("nonpolymer_entity_instances") or []:
                rcsb_id = instance.get("rcsb_id")
                validation_score = instance.get(
                    "rcsb_nonpolymer_instance_validation_score"
                )
                if (
                    validation_score
                    and isinstance(validation_score, list)
                    and validation_score[0]
                ):
                    ranking_model_fit = validation_score[0].get("ranking_model_fit")
                    if ranking_model_fit is not None:
                        extracted_data[rcsb_id] = ranking_model_fit

        return extracted_data

    except (KeyError, TypeError, ValueError) as e:
        logger.warning("Error processing response for %s: %s", pdb_id, e)
        return {}
