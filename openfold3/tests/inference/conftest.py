"""Shared fixtures for the inference integration tests.

These tests reach real external services (the ColabFold MSA server and, through the
template chain-ID remap, the RCSB PDB GraphQL API). The RCSB fixture below keeps the
number of RCSB round trips per session to one per distinct set of PDB IDs.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import openfold3.core.data.tools.colabfold_msa_server as colabfold_msa_server


@pytest.fixture(scope="session", autouse=True)
def cache_rcsb_chain_id_mappings() -> Iterator[dict[frozenset[str], dict]]:
    """Memoise ``fetch_label_to_author_chain_ids`` for the whole session.

    Several parametrized cases submit the same sequence (e.g. ubiquitin, whose ~200
    ColabFold template hits make the largest chain-mapping query in the suite), so
    without this every case re-asks RCSB for the same answer. The cache is keyed on
    the exact ID set, so a case with different template hits still fetches its own.

    The function is patched where ``colabfold_msa_server`` bound it, since that
    module imported the name directly. Failures are not cached: a timed-out fetch is
    retried by the next case that needs it.
    """
    real_fetch = colabfold_msa_server.fetch_label_to_author_chain_ids
    cache: dict[frozenset[str], dict] = {}

    def cached_fetch(pdb_ids: set[str]) -> dict[str, dict[str, str]]:
        key = frozenset(pdb_ids)
        if key not in cache:
            cache[key] = real_fetch(pdb_ids)
        return cache[key]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            colabfold_msa_server, "fetch_label_to_author_chain_ids", cached_fetch
        )
        yield cache
