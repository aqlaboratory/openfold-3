"""All operations for processing and manipulating metadata and training caches."""

from datetime import date, datetime
from typing import TypeAlias

StructureMetadataCache: TypeAlias = dict[str, dict]
"""
Type alias for the "structure_metadata" part of the preprocessed metadata_cache.json.

Follows the format:
{
    ...,
    pdb_id: {
        ...basic_metadata...
        "chains": {...}
        "interfaces": [...]
    }
    pdb_id: {...}
}
"""


def filter_by_release_date(
    cache: StructureMetadataCache, max_date: date | str
) -> StructureMetadataCache:
    """Filter the cache by removing entries newer than a given date.

    Args:
        max_date:
            Maximum date that the PDB entry was released to be included in the cache.
        cache:
            The cache to filter.

    Returns:
        The filtered cache.
    """
    if not isinstance(max_date, date):
        max_date = datetime.strptime(max_date, "%Y-%m-%d").date()

    cache = {
        pdb_id: metadata
        for pdb_id, metadata in cache.items()
        if datetime.strptime(metadata["release_date"], "%Y-%m-%d").date() <= max_date
    }

    return cache


def filter_by_resolution(
    cache: StructureMetadataCache, max_resolution: float
) -> StructureMetadataCache:
    """Filter the cache by removing entries with resolution higher than a given value.

    Args:
        cache:
            The cache to filter.
        max_resolution:
            Filter out entries with resolution (numerically) higher than this value.
            E.g. if max_resolution=9.0, entries with resolution 9.1 Å or higher will be
            removed.

    Returns:
        The filtered cache.
    """
    cache = {
        pdb_id: metadata
        for pdb_id, metadata in cache.items()
        if metadata["resolution"] <= max_resolution
    }

    return cache


def chain_cache_entry_is_polymer(entry: dict) -> bool:
    """Check if the entry of a particular chain in the metadata cache is a polymer."""
    return entry["molecule_type"] in ("PROTEIN", "DNA", "RNA")


def filter_by_max_polymer_chains(
    cache: StructureMetadataCache, max_chains: int
) -> StructureMetadataCache:
    """Filter the cache by removing entries with more polymer chains than a given value.

    Args:
        cache:
            The cache to filter.
        max_chains:
            Filter out entries with more polymer chains than this value.

    Returns:
        The filtered cache.
    """

    cache = {
        pdb_id: metadata
        for pdb_id, metadata in cache.items()
        if sum(
            chain_cache_entry_is_polymer(chain) for chain in metadata["chains"].values()
        )
        <= max_chains
    }

    return cache


def filter_by_skipped_structures(
    cache: StructureMetadataCache,
) -> StructureMetadataCache:
    """Filter the cache by removing entries that were skipped during preprocessing.

    Args:
        cache:
            The cache to filter.

    Returns:
        The filtered cache.
    """
    cache = {
        pdb_id: metadata
        for pdb_id, metadata in cache.items()
        if metadata["status"] == "success"
    }

    return cache


def map_chains_to_representatives(
    query_seq_dict: dict[str, str], repr_seq_dict: dict[str, str]
) -> dict[str, str]:
    """Maps chains to their representative chains.

    This takes in a dictionary of query IDs and sequences and a similar dictionary of
    representative IDs and sequences and maps the query chains to a representative with
    the same sequence. This information is necessary for the training cache as MSA
    databases are usually deduplicated.

    Args:
        query_seq_dict:
            Dictionary mapping chain IDs to sequences.
        repr_seq_dict:
            Dictionary mapping chain IDs to sequences.

    Returns:
        Dictionary mapping query chain IDs to representative chain IDs.
    """

    # Convert to seq -> chain mapping for easier lookup
    repr_seq_to_chain = {seq: chain for chain, seq in repr_seq_dict.items()}

    query_to_repr = {}

    # Map each query chain to its representative
    for query_chain, query_seq in query_seq_dict.items():
        repr_chain = repr_seq_to_chain.get(query_seq)

        query_to_repr[query_chain] = repr_chain

    return query_to_repr


def add_chain_representatives(
    cache: StructureMetadataCache,
    query_chain_to_seq: dict[str, str],
    repr_chain_to_seq: dict[str, str],
) -> None:
    """Add alignment representatives to the structure metadata cache.

    Will find the representative chain for each query chain and add it to the cache
    in-place under a new "alignment_representative_id" key for each chain.

    Args:
        cache:
            The structure metadata cache to update.
        query_chain_to_seq:
            Dictionary mapping query chain IDs to sequences.
        repr_chain_to_seq:
            Dictionary mapping representative chain IDs to sequences.
    """
    query_chains_to_repr_chains = map_chains_to_representatives(
        query_chain_to_seq, repr_chain_to_seq
    )

    for pdb_id, metadata in cache.items():
        for chain_id, chain_metadata in metadata["chains"].items():
            repr_id = query_chains_to_repr_chains.get(f"{pdb_id}_{chain_id}")

            chain_metadata["alignment_representative_id"] = repr_id


def filter_no_alignment_representative(cache: StructureMetadataCache):
    """Filter the cache by removing entries with no alignment representative.

    If any of the chains in the entry do not have corresponding alignment data, the
    entire entry is removed from the cache.

    Args:
        cache:
            The cache to filter.

    Returns:
        The filtered cache.
    """
    filtered_cache = {}

    for pdb_id, metadata in cache.items():
        # Add only entries to filtered cache where all protein or RNA chains have
        # alignment representatives
        for chain in metadata["chains"].values():
            if chain["molecule_type"] not in ("PROTEIN", "RNA"):
                continue

            if chain["alignment_representative_id"] is None:
                break
        else:
            filtered_cache[pdb_id] = metadata

    return filtered_cache
