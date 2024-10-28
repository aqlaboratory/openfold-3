"""All operations for processing and manipulating metadata and training caches."""

import functools
import logging
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from openfold3.core.data.io.sequence.fasta import (
    read_multichain_fasta,
)
from openfold3.core.data.primitives.dataset_cache.types import (
    ChainData,
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    ClusteredDatasetStructureDataCache,
    DatasetReferenceMoleculeCache,
    DatasetReferenceMoleculeData,
    PreprocessingDataCache,
    PreprocessingStructureDataCache,
    StructureDataCache,
)
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger(__name__)


def func_with_n_filtered_chain_log(
    structure_cache_filter_func: callable, logger: logging.Logger
) -> None:
    """Decorator to log the number of chains removed by a structure cache filter func.

    Args:
        structure_cache_filter_func:
            The filter function to apply to a structure data cache.

    Returns:
        The decorated function that logs the number of chains removed.
    """

    @functools.wraps(structure_cache_filter_func)
    def wrapper(
        structure_cache: StructureDataCache, *args, **kwargs
    ) -> StructureDataCache:
        # Note that this doesn't count skipped/failed structures for which we have no
        # number of chain information
        num_chains_before = sum(
            len(metadata.chains) if metadata.chains else 0
            for metadata in structure_cache.values()
        )

        output = structure_cache_filter_func(structure_cache, *args, **kwargs)

        if isinstance(output, tuple):
            structure_cache = output[0]

            if not isinstance(structure_cache, dict):
                raise ValueError(
                    "The first element of the output tuple must be a "
                    + "StructureDataCache."
                )
        else:
            structure_cache = output

            if not isinstance(structure_cache, dict):
                raise ValueError("The output must be a StructureDataCache.")

        num_chains_after = sum(
            len(metadata.chains) if metadata.chains else 0
            for metadata in structure_cache.values()
        )

        num_chains_removed = num_chains_before - num_chains_after
        percentage_removed = (num_chains_removed / num_chains_before) * 100

        logger.info(
            f"Function {structure_cache_filter_func.__name__} removed "
            + f"{num_chains_removed} chains ({percentage_removed:.2f}%)."
        )

        return output

    return wrapper


def filter_by_release_date(
    structure_cache: StructureDataCache,
    max_date: date | str,
) -> StructureDataCache:
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

    structure_cache = {
        pdb_id: metadata
        for pdb_id, metadata in structure_cache.items()
        if metadata.release_date <= max_date
    }

    return structure_cache


def filter_by_resolution(
    structure_cache: StructureDataCache,
    max_resolution: float,
) -> StructureDataCache:
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
    structure_cache = {
        pdb_id: metadata
        for pdb_id, metadata in structure_cache.items()
        if metadata.resolution <= max_resolution
    }

    return structure_cache


def chain_cache_entry_is_polymer(
    chain_data: ChainData,
) -> bool:
    """Check if the entry of a particular chain in the metadata cache is a polymer."""
    return chain_data.molecule_type in (
        MoleculeType.PROTEIN,
        MoleculeType.DNA,
        MoleculeType.RNA,
    )


def filter_by_max_polymer_chains(
    structure_cache: StructureDataCache,
    max_chains: int,
) -> StructureDataCache:
    """Filter the cache by removing entries with more polymer chains than a given value.

    Args:
        cache:
            The cache to filter.
        max_chains:
            Filter out entries with more polymer chains than this value.

    Returns:
        The filtered cache.
    """
    # Refactor accounting for previously defined dataclass
    structure_cache = {
        pdb_id: structure_data
        for pdb_id, structure_data in structure_cache.items()
        if sum(
            chain_cache_entry_is_polymer(chain)
            for chain in structure_data.chains.values()
        )
        <= max_chains
    }

    return structure_cache


def filter_by_skipped_structures(
    structure_cache: PreprocessingStructureDataCache,
) -> PreprocessingStructureDataCache:
    """Filter the cache by removing entries that were skipped during preprocessing.

    Args:
        cache:
            The cache to filter.

    Returns:
        The filtered cache.
    """
    structure_cache = {
        pdb_id: metadata
        for pdb_id, metadata in structure_cache.items()
        if metadata.status == "success"
    }

    return structure_cache


def build_provisional_clustered_dataset_cache(
    preprocessing_cache: PreprocessingDataCache, dataset_name: str
) -> ClusteredDatasetCache:
    """Build a preliminary clustered-dataset cache with empty new values.

    Reformats the PreprocessingDataCache to the ClusteredDatasetCache format, with empty
    values for the new fields that will be filled in later.

    Args:
        preprocessing_cache:
            The cache to convert.
        dataset_name:
            The name that the dataset should be referred to as.

    Returns:
        The new cache with a mixture of previous fields and new fields with empty
        placeholder values.
    """
    structure_data = {}
    reference_molecule_data = {}

    prepr_structure_data = preprocessing_cache.structure_data

    # First create structure data
    for pdb_id, preprocessed_structure_data in prepr_structure_data.items():
        structure_data[pdb_id] = ClusteredDatasetStructureData(
            release_date=preprocessed_structure_data.release_date,
            resolution=preprocessed_structure_data.resolution,
            chains={},
            interfaces={},
        )

        # Add all the chain metadata with dummy cluster values
        new_chain_data = structure_data[pdb_id].chains
        for chain_id, chain_data in preprocessed_structure_data.chains.items():
            new_chain_data[chain_id] = ClusteredDatasetChainData(
                label_asym_id=chain_data.label_asym_id,
                auth_asym_id=chain_data.auth_asym_id,
                entity_id=chain_data.entity_id,
                molecule_type=chain_data.molecule_type,
                reference_mol_id=chain_data.reference_mol_id,
                cluster_id="",
                cluster_size=0,
                alignment_representative_id=None,
            )

        # Add interface cluster data with dummy values
        new_interface_clusters = structure_data[pdb_id].interface_clusters
        for interface in preprocessed_structure_data.interfaces:
            chain_1, chain_2 = interface
            interface_id = f"{chain_1}_{chain_2}"
            new_interface_clusters[interface_id] = ClusteredDatasetInterfaceData(
                cluster_id="",
                cluster_size=0,
            )

    # Create reference molecule data with set_fallback_to_nan=False everywhere (for now)
    prepr_ref_mol_data = preprocessing_cache.reference_molecule_data

    for ref_mol_id, ref_mol_data in prepr_ref_mol_data.items():
        reference_molecule_data[ref_mol_id] = DatasetReferenceMoleculeData(
            conformer_gen_strategy=ref_mol_data.conformer_gen_strategy,
            fallback_conformer_pdb_id=ref_mol_data.fallback_conformer_pdb_id,
            canonical_smiles=ref_mol_data.canonical_smiles,
            set_fallback_to_nan=False,
        )

    new_dataset_cache = ClusteredDatasetCache(
        name=dataset_name,
        structure_data=structure_data,
        reference_molecule_data=reference_molecule_data,
    )
    return new_dataset_cache


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
    structure_cache: ClusteredDatasetStructureDataCache,
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

    for pdb_id, metadata in structure_cache.items():
        for chain_id, chain_metadata in metadata.chains.items():
            repr_id = query_chains_to_repr_chains.get(f"{pdb_id}_{chain_id}")

            chain_metadata.alignment_representative_id = repr_id


def filter_no_alignment_representative(
    structure_cache: ClusteredDatasetStructureDataCache, return_no_repr=False
) -> (
    ClusteredDatasetStructureDataCache
    | tuple[ClusteredDatasetStructureDataCache, dict[str, ClusteredDatasetChainData]]
):
    """Filter the cache by removing entries with no alignment representative.

    If any of the chains in the entry do not have corresponding alignment data, the
    entire entry is removed from the cache.

    Args:
        cache:
            The cache to filter.
        return_no_repr:
            If True, also return a dictionary of unmatched entries, formatted as:
            pdb_id: chain_metadata

            Note that this is a subset of all effectively removed chains, as even a
            single unmatched chain will result in exclusion of the entire PDB structure.
            Default is False.

    Returns:
        The filtered cache, or the filtered cache and the unmatched entries if
        return_no_repr is True.
    """
    filtered_cache = {}

    if return_no_repr:
        unmatched_entries = defaultdict(dict)

    for pdb_id, metadata in structure_cache.items():
        all_in_cache_have_repr = True

        # Add only entries to filtered cache where all protein or RNA chains have
        # alignment representatives
        for chain_id, chain_data in metadata.chains.items():
            if chain_data.molecule_type not in (MoleculeType.PROTEIN, MoleculeType.RNA):
                continue

            if chain_data.alignment_representative_id is None:
                all_in_cache_have_repr = False

                # If return_removed is True, also try finding remaining chains with no
                # alignment representative, otherwise break early
                if return_no_repr:
                    unmatched_entries[pdb_id][chain_id] = chain_data
                else:
                    break

        if all_in_cache_have_repr:
            filtered_cache[pdb_id] = metadata

    if return_no_repr:
        return filtered_cache, unmatched_entries
    else:
        return filtered_cache


def add_and_filter_alignment_representatives(
    structure_cache: ClusteredDatasetStructureDataCache,
    query_chain_to_seq: dict[str, str],
    alignment_representatives_fasta: Path,
    return_no_repr=False,
) -> (
    ClusteredDatasetStructureDataCache
    | tuple[ClusteredDatasetStructureDataCache, dict[str, ClusteredDatasetChainData]]
):
    """Adds alignment representatives to cache and filters out entries without any.

    Will find the representative chain for each query chain and add it to the cache
    in-place under a new "alignment_representative_id" key for each chain. Entries
    without alignment representatives are removed from the cache.

    Args:
        cache:
            The structure metadata cache to update.
        alignment_representatives_fasta:
            Path to the FASTA file containing alignment representatives.
        query_chain_to_seq:
            Dictionary mapping query chain IDs to sequences.
        return_no_repr:
            If True, also return a dictionary of unmatched entries, formatted as:
            pdb_id: chain_metadata

            Default is False.

    Returns:
        The filtered cache, or the filtered cache and the unmatched entries if
        return_no_repr is True.
    """
    repr_chain_to_seq = read_multichain_fasta(alignment_representatives_fasta)
    add_chain_representatives(structure_cache, query_chain_to_seq, repr_chain_to_seq)

    if return_no_repr:
        structure_cache, unmatched_entries = filter_no_alignment_representative(
            structure_cache, return_no_repr=True
        )
        return structure_cache, unmatched_entries
    else:
        structure_cache = filter_no_alignment_representative(structure_cache)
        return structure_cache


def get_all_cache_chains(
    structure_cache: StructureDataCache,
) -> set[str]:
    """Get all chain IDs in the cache.

    Args:
        cache:
            The cache to get chains from.

    Returns:
        A set of all chain IDs in the cache.
    """
    all_chains = set()

    for pdb_id, metadata in structure_cache.items():
        for chain_id in metadata.chains:
            all_chains.add(f"{pdb_id}_{chain_id}")

    return all_chains


def set_nan_fallback_conformer_flag(
    pdb_id_to_release_date: dict[str, date | str],
    reference_mol_cache: DatasetReferenceMoleculeCache,
    max_model_pdb_release_date: date | str,
) -> None:
    """Set the fallback conformer to NaN for ref-coordinates from PDB IDs after a cutoff

    Based on AF3 SI 2.8, fallback conformers derived from PDB coordinates cannot be used
    if the corresponding PDB model was released after the training cutoff. This function
    introduces a new key "set_fallback_to_nan" in the reference molecule cache, which is
    set to True for these cases and will be read in the model dataloading pipeline.

    Args:
        structure_cache:
            The structure metadata cache.
        reference_mol_cache:
            The reference molecule metadata cache.
        max_pdb_date:
            The maximum PDB release date for structures in the training set. PDB IDs
            released after this date will have their fallback conformer set to NaN.

    """
    if not isinstance(max_model_pdb_release_date, date):
        max_model_pdb_release_date = datetime.strptime(
            max_model_pdb_release_date, "%Y-%m-%d"
        ).date()

    for ref_mol_id, metadata in reference_mol_cache.items():
        # Check if the fallback conformer should be NaN
        model_pdb_id = metadata.fallback_conformer_pdb_id

        if model_pdb_id is None:
            continue

        elif model_pdb_id not in pdb_id_to_release_date:
            logger.warning(
                f"Fallback fonformer PDB ID {model_pdb_id} not found in cache, for "
                f"molecule {ref_mol_id}, forcing NaN fallback conformer."
            )
        # Check if the PDB ID's release date is after the cutoff
        elif pdb_id_to_release_date[model_pdb_id] > max_model_pdb_release_date:
            logger.debug(f"Setting fallback conformer to NaN for {ref_mol_id}.")
            metadata.set_fallback_to_nan = True
        else:
            metadata.set_fallback_to_nan = False

    return None
