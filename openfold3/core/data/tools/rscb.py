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

import requests

logger = logging.getLogger(__name__)

_RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

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


def fetch_label_to_author_chain_ids(
    pdb_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Fetch label-to-author chain ID mappings from the RCSB PDB GraphQL API.

    Makes a single batched request for all PDB IDs and returns a nested dict
    mapping ``entry_id`` → ``label_asym_id`` → ``author_chain_id``.

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

    try:
        resp = requests.post(
            _RCSB_GRAPHQL_URL,
            json={
                "query": _CHAIN_MAPPING_QUERY,
                "variables": {"ids": sorted(pdb_ids)},
            },
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch chain ID mappings from RCSB for "
            f"{len(pdb_ids)} entries. Cannot proceed without chain ID "
            f"re-mapping."
        ) from e

    data = resp.json().get("data", {})
    entries = data.get("entries") or []

    result: dict[str, dict[str, str]] = {}
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
        timeout=30,
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
