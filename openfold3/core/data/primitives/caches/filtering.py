"""All operations for processing and manipulating metadata and training caches."""

import functools
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple, Union

import requests
from tqdm import tqdm

from openfold3.core.data.io.sequence.fasta import (
    read_multichain_fasta,
)
from openfold3.core.data.primitives.caches.format import (
    ChainData,
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    ClusteredDatasetStructureDataCache,
    DatasetCache,
    DatasetReferenceMoleculeCache,
    DatasetReferenceMoleculeData,
    PreprocessingDataCache,
    PreprocessingStructureDataCache,
    StructureDataCache,
    ValidationDatasetCache,
    ValidationDatasetChainData,
    ValidationDatasetInterfaceData,
    ValidationDatasetReferenceMoleculeData,
    ValidationDatasetStructureData,
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


def filter_by_token_count(
    structure_cache: StructureDataCache,
    max_tokens: int,
) -> StructureDataCache:
    """Filter the cache by removing entries with token count higher than a given value.

    Args:
        structure_cache:
            The structure cache to filter.
        max_tokens:
            Filter out entries with token count higher than this value.
    """
    structure_cache = {
        pdb_id: metadata
        for pdb_id, metadata in structure_cache.items()
        if metadata.token_count <= max_tokens
    }

    return structure_cache


def filter_by_release_date(
    structure_cache: StructureDataCache,
    min_date: Union[date | str] | None = None,
    max_date: Union[date | str] | None = None,
) -> StructureDataCache:
    """Filters the cache to only include entries within a specified date range.

    Filters the cache to only include entries whose release_date is within the specified
    [min_date, max_date] range. Supports None for either or both min_date/max_date, in
    which case that bound is ignored.

    Args:
        structure_cache (StructureDataCache):
            The structure cache to filter.
        min_date (date | str | None):
            Minimum release date (inclusive). If None, no lower bound is applied.
        max_date (date | str | None):
            Maximum release date (inclusive). If None, no upper bound is applied.

    Returns:
        The filtered cache containing only entries matching the specified date range.
    """
    # Convert min_date to date if it's a string
    if isinstance(min_date, str):
        min_date = datetime.strptime(min_date, "%Y-%m-%d").date()

    # Convert max_date to date if it's a string
    if isinstance(max_date, str):
        max_date = datetime.strptime(max_date, "%Y-%m-%d").date()

    filtered_cache = {
        pdb_id: metadata
        for pdb_id, metadata in structure_cache.items()
        if (min_date is None or metadata.release_date >= min_date)
        and (max_date is None or metadata.release_date <= max_date)
    }

    return filtered_cache


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


# NIT: Make this class-method of ClusteredDataset instead?
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
                cluster_id=None,
                cluster_size=None,
                alignment_representative_id=None,
                template_ids=None,  # added in a separate script after
            )

        # Add interface cluster data with dummy values
        new_interface_data = structure_data[pdb_id].interfaces
        for interface in preprocessed_structure_data.interfaces:
            chain_1, chain_2 = interface
            interface_id = f"{chain_1}_{chain_2}"
            new_interface_data[interface_id] = ClusteredDatasetInterfaceData(
                cluster_id=None,
                cluster_size=None,
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


# TODO: This is too redundant with the previous function, but also the build_provisional
# logic in general might not be the best way to go about this
def build_provisional_clustered_val_dataset_cache(
    preprocessing_cache: PreprocessingDataCache, dataset_name: str
) -> ValidationDatasetCache:
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
        structure_data[pdb_id] = ValidationDatasetStructureData(
            release_date=preprocessed_structure_data.release_date,
            resolution=preprocessed_structure_data.resolution,
            token_count=preprocessed_structure_data.token_count,
            chains={},
            interfaces={},
        )

        # Add all the chain metadata with dummy cluster values
        new_chain_data = structure_data[pdb_id].chains
        for chain_id, chain_data in preprocessed_structure_data.chains.items():
            new_chain_data[chain_id] = ValidationDatasetChainData(
                label_asym_id=chain_data.label_asym_id,
                auth_asym_id=chain_data.auth_asym_id,
                entity_id=chain_data.entity_id,
                molecule_type=chain_data.molecule_type,
                reference_mol_id=chain_data.reference_mol_id,
                alignment_representative_id=None,
                template_ids=None,
                cluster_id=None,
                cluster_size=None,
                low_homology=None,
                use_metrics=False,
                ranking_model_fit=None,
            )

        # Add interface cluster data with dummy values
        new_interface_data = structure_data[pdb_id].interfaces
        for interface in preprocessed_structure_data.interfaces:
            chain_1, chain_2 = interface
            interface_id = f"{chain_1}_{chain_2}"
            new_interface_data[interface_id] = ValidationDatasetInterfaceData(
                cluster_id=None,
                cluster_size=None,
                low_homology=None,
                metric_eligible=None,
                use_metrics=False,
            )

    # Create reference molecule data with set_fallback_to_nan=False everywhere (for now)
    prepr_ref_mol_data = preprocessing_cache.reference_molecule_data

    for ref_mol_id, ref_mol_data in prepr_ref_mol_data.items():
        reference_molecule_data[ref_mol_id] = ValidationDatasetReferenceMoleculeData(
            conformer_gen_strategy=ref_mol_data.conformer_gen_strategy,
            fallback_conformer_pdb_id=ref_mol_data.fallback_conformer_pdb_id,
            canonical_smiles=ref_mol_data.canonical_smiles,
            residue_count=ref_mol_data.residue_count,
            set_fallback_to_nan=False,
        )

    new_dataset_cache = ValidationDatasetCache(
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
    restrict_to_molecule_types: list[MoleculeType] | None = None,
) -> set[str]:
    """Get all chain IDs in the cache.

    Args:
        cache:
            The cache to get chains from.
        restrict_molecule_type:
            If not None, only return chains of this molecule type.

    Returns:
        A set of all chain IDs in the cache.
    """
    all_chains = set()

    for pdb_id, metadata in structure_cache.items():
        for chain_id in metadata.chains:
            if (
                restrict_to_molecule_types is None
                or metadata.chains[chain_id].molecule_type in restrict_to_molecule_types
            ):
                all_chains.add(f"{pdb_id}_{chain_id}")

    return all_chains


def get_mol_id_to_smiles(
    dataset_cache: DatasetCache,
) -> dict[str, str]:
    """Get mapping from molecule IDs to SMILES strings for all ligands in the cache."""
    structure_cache = dataset_cache.structure_data
    ref_mol_cache = dataset_cache.reference_molecule_data

    mol_id_to_smiles = {}

    for structure_data in structure_cache.values():
        for chain_data in structure_data.chains.values():
            if chain_data.molecule_type == MoleculeType.LIGAND:
                mol_id = chain_data.reference_mol_id
                smiles = ref_mol_cache[mol_id].canonical_smiles
                mol_id_to_smiles[mol_id] = smiles

    return mol_id_to_smiles


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


# TODO: Do this in preprocessing instead to avoid it going out-of-sync with the data?
def get_model_ranking_fit(pdb_id):
    """Fetches the model ranking fit entries for all ligands of a single PDB-ID.

    Uses the PDB GraphQL API to fetch the model ranking fit values for all ligands in a
    single PDB entry. Note that this function will always fetch from the newest version
    of the PDB and can therefore occasionally give incorrect results for old datasets
    whose structures have been updated since.
    """
    url = "https://data.rcsb.org/graphql"  # RCSB PDB's GraphQL API endpoint

    query = """
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

    # Prepare the request with the pdb_id as a variable
    variables = {"pdb_id": pdb_id}

    # Make the request to the GraphQL endpoint using the variables
    response = requests.post(url, json={"query": query, "variables": variables})

    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Parse the JSON response
            data = response.json()

            # Safely navigate through data
            entry_data = data.get("data", {}).get("entry", {})
            if not entry_data:
                return {}

            extracted_data = {}

            # Check for nonpolymer_entities
            nonpolymer_entities = entry_data.get("nonpolymer_entities", [])

            if nonpolymer_entities:
                for entity in nonpolymer_entities:
                    for instance in entity.get("nonpolymer_entity_instances", []):
                        rcsb_id = instance.get("rcsb_id")
                        validation_score = instance.get(
                            "rcsb_nonpolymer_instance_validation_score"
                        )

                        if (
                            validation_score
                            and isinstance(validation_score, list)
                            and validation_score[0]
                        ):
                            ranking_model_fit = validation_score[0].get(
                                "ranking_model_fit"
                            )
                            if ranking_model_fit is not None:
                                extracted_data[rcsb_id] = ranking_model_fit

            return extracted_data

        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing response for {pdb_id}: {e}")
            return {}
    else:
        print(f"Request failed with status code {response.status_code}")
        return {}


def assign_ligand_model_fits(
    structure_cache: ValidationDatasetCache, num_threads: int = 3
) -> None:
    """Fetch the model ranking fit values for all ligands in the cache.

    Will add the "ranking_model_fit" field to all ligand chains in the cache, with the
    corresponding model ranking fit value.

    Args:
        structure_cache:
            The cache to fetch model fit values for.
        num_threads:
            The number of threads to use for fetching the model fit values. Default is
            3 due to PDB-API rate limits.

    Returns:
        None, the structure cache is updated in-place.
    """

    def fetch_ligand_model_fits(
        pdb_id: str, structure_data: ValidationDatasetStructureData
    ):
        """Add the ranking_model_fit values for a single PDB entry."""
        ligand_chain_data = [
            chain_data
            for chain_data in structure_data.chains.values()
            if chain_data.molecule_type == MoleculeType.LIGAND
        ]

        if len(ligand_chain_data) == 0:
            return

        ligand_fits = get_model_ranking_fit(pdb_id)

        # Filter chains
        for chain in ligand_chain_data:
            rcsb_id = f"{pdb_id.upper()}.{chain.label_asym_id}"

            # Set to worst possible fit if not found
            chain.ranking_model_fit = ligand_fits.get(rcsb_id, 0.0)

    if num_threads == 1:
        for pdb_id, structure_data in tqdm(structure_cache.items()):
            fetch_ligand_model_fits(pdb_id, structure_data)
    else:
        # Use threading to speed up the queries
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            all_pdb_ids = list(structure_cache.keys())
            all_pdb_data = list(structure_cache.values())

            list(
                tqdm(
                    executor.map(fetch_ligand_model_fits, all_pdb_ids, all_pdb_data),
                    total=len(structure_cache),
                    desc="Fetching model fit values",
                )
            )


class InterfaceDataPoint(NamedTuple):
    """Specifies a single interface in a dataset cache."""

    pdb_id: str
    interface_id: str


def filter_cache_by_specified_interfaces(
    dataset_cache: DatasetCache, keep_interface_datapoints: set[InterfaceDataPoint]
) -> None:
    """In-place deletes the chains and interfaces not specified to be kept.

    Will remove all interfaces not in the pdb_id_to_keep_interfaces dictionary, and all
    chains that are not part of those interfaces. Will additionally remove the PDB
    entirely if no chains or interfaces are kept.

    Args:
        dataset_cache (DatasetCache):
            The cache to remove chains and interfaces from.
        keep_interface_datapoints (set[InterfaceDataPoint]):
            A set of (pdb_id, interface_id) tuples specifying which interfaces to keep.


    Returns:
        None, the cache is updated in-place.
    """
    # Create dictionary mapping each PDB-ID to all interfaces to keep
    pdb_id_to_keep_interfaces = defaultdict(set)
    for pdb_id, interface_id in keep_interface_datapoints:
        pdb_id_to_keep_interfaces[pdb_id].add(interface_id)

    # Delete structures with no interface to keep at all
    all_pdb_ids_cache = set(dataset_cache.structure_data.keys())
    pdb_ids_to_remove = all_pdb_ids_cache - set(pdb_id_to_keep_interfaces.keys())

    for pdb_id in pdb_ids_to_remove:
        del dataset_cache.structure_data[pdb_id]

    # Delete anything not in specified interfaces
    for pdb_id, structure_data in dataset_cache.structure_data.items():
        interfaces_to_keep = pdb_id_to_keep_interfaces[pdb_id]
        chains_to_keep = set(
            chain_id
            for interface_id in interfaces_to_keep
            for chain_id in interface_id.split("_")
        )
        interfaces_to_remove = (
            set(structure_data.interfaces.keys()) - interfaces_to_keep
        )
        chains_to_remove = set(structure_data.chains.keys()) - chains_to_keep

        for interface_id in interfaces_to_remove:
            del structure_data.interfaces[interface_id]
        for chain_id in chains_to_remove:
            del structure_data.chains[chain_id]

    # TODO: Remove at some point
    assert not any(
        len(structure_data.chains) == 0 or len(structure_data.interfaces) == 0
        for structure_data in dataset_cache.structure_data.values()
    )


def subsample_interfaces_per_cluster(
    dataset_cache: DatasetCache,
    num_interfaces_per_cluster: int = 1,
    random_seed: int | None = None,
) -> None:
    """Subsamples a fixed number of interfaces per cluster.

    Will subsample a fixed number of interfaces per cluster, keeping only those
    interfaces in the cache.

    Args:
        dataset_cache (DatasetCache):
            The cache to subsample.
        num_interfaces_per_cluster (int):
            The number of interfaces to keep per cluster. Default is 1.
        random_seed (int | None):
            The random seed to use for sampling. Default is None.

    Returns:
        None, the cache is updated in-place.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Get all interface datapoints belonging to each cluster
    cluster_id_to_interfaces = defaultdict(list)
    for pdb_id, structure_data in dataset_cache.structure_data.items():
        for interface_id in structure_data.interfaces:
            cluster_id = structure_data.interfaces[interface_id].cluster_id
            cluster_id_to_interfaces[cluster_id].append(
                InterfaceDataPoint(pdb_id, interface_id)
            )

    # Take specified number of interfaces per cluster
    subsampled_interface_datapoints = []
    for interface_datapoints in cluster_id_to_interfaces.values():
        subsampled_interface_datapoints.extend(
            random.sample(interface_datapoints, num_interfaces_per_cluster)
        )

    filter_cache_by_specified_interfaces(
        dataset_cache, set(subsampled_interface_datapoints)
    )


class ChainDataPoint(NamedTuple):
    """Specifies a single chain in a dataset cache."""

    pdb_id: str
    chain_id: str


def filter_cache_to_specified_chains(
    dataset_cache: DatasetCache, keep_chain_datapoints: set[ChainDataPoint]
) -> None:
    """In-place deletes the chains and interfaces not specified to be kept.

    This code deletes all interfaces and will only keep the specified chains.

    Args:
        dataset_cache (DatasetCache):
            The cache to remove chains and interfaces from.
        keep_chain_datapoints (set[ChainDataPoint]):
            A set of (pdb_id, chain_id) tuples specifying which chains to keep.

    Returns:
        None, the cache is updated in-place(!)
    """
    # Create dictionary mapping each PDB-ID to all chains to keep
    pdb_id_to_keep_chains = defaultdict(set)
    for pdb_id, chain_id in keep_chain_datapoints:
        pdb_id_to_keep_chains[pdb_id].add(chain_id)

    # Delete all PDBs that have no chains to keep
    all_pdb_ids_cache = set(dataset_cache.structure_data.keys())
    pdb_ids_to_remove = all_pdb_ids_cache - set(pdb_id_to_keep_chains.keys())

    for pdb_id in pdb_ids_to_remove:
        del dataset_cache.structure_data[pdb_id]

    # Delete all chains not in the specified set
    for pdb_id, structure_data in dataset_cache.structure_data.items():
        chains_to_keep = pdb_id_to_keep_chains[pdb_id]
        chains_to_remove = set(structure_data.chains.keys()) - chains_to_keep

        for chain_id in chains_to_remove:
            del structure_data.chains[chain_id]

        # Set interfaces to empty dict
        structure_data.interfaces = dict()


def subsample_chains_by_type(
    dataset_cache: ClusteredDatasetCache,
    n_protein: int | None = 40,
    n_dna: int | None = None,
    n_rna: int | None = None,
    random_seed: int | None = None,
) -> None:
    """Selects a fixed number of chains by molecule type.

    Follows AF3 SI 5.8 Monomer Selection Step 4). The function subsamples specific
    chains and deletes all other chains from the cache.

    Note that proteins are sampled as unique cluster representatives, which is not
    directly stated in the SI but seems logical given that chains are preclustered.

    Args:
        dataset_cache (ClusteredDatasetCache):
            The cache to subsample.
        n_protein (int | None):
            The number of protein chains to sample. Default is 40.
        n_dna (int | None):
            The number of DNA chains to sample. Default is None, which means that all
            DNA chains will be selected across all clusters.
        n_rna (int | None):
            The number of RNA chains to sample. Default is None, which means that all
            RNA chains will be selected across all clusters.
        random_seed (int | None):
            The random seed to use for sampling. Default is None.

    Returns:
        None, the cache is updated in-place.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Store the chain data points grouped by cluster
    chain_type_to_clusters = {
        MoleculeType.PROTEIN: defaultdict(list),
        MoleculeType.DNA: defaultdict(list),
        MoleculeType.RNA: defaultdict(list),
    }
    chain_type_to_n_samples = {
        MoleculeType.PROTEIN: n_protein,
        MoleculeType.DNA: n_dna,
        MoleculeType.RNA: n_rna,
    }

    # Collect all chain data points of the specified types grouped by cluster
    for pdb_id, structure_data in dataset_cache.structure_data.items():
        for chain_id, chain_data in structure_data.chains.items():
            chain_type = chain_data.molecule_type

            if chain_type not in chain_type_to_clusters:
                continue

            chain_type_to_clusters[chain_type][chain_data.cluster_id].append(
                ChainDataPoint(pdb_id, chain_id)
            )

    keep_chain_datapoints = set()

    # Subsample the chains, taking one per cluster except if the count is set to None in
    # which case all samples are taken
    for chain_type, clusters in chain_type_to_clusters.items():
        n_samples = chain_type_to_n_samples[chain_type]

        # Take every single datapoint if n_samples is None
        if n_samples is None:
            for chain_datapoints in clusters.values():
                keep_chain_datapoints.update(chain_datapoints)

        # Otherwise, take 1 sample from n_samples clusters
        else:
            sampled_clusters = random.sample(list(clusters.keys()), n_samples)

            for cluster_id in sampled_clusters:
                keep_chain_datapoints.add(random.choice(clusters[cluster_id]))

    # Remove everything outside of the selected chains
    filter_cache_to_specified_chains(dataset_cache, keep_chain_datapoints)


def subsample_interfaces_by_type(
    dataset_cache: DatasetCache,
    n_protein_protein: int | None = 600,
    n_protein_dna: int | None = 100,
    n_dna_dna: int | None = 100,
    n_protein_ligand: int | None = 600,
    n_dna_ligand: int | None = 50,
    n_ligand_ligand: int | None = 200,
    n_protein_rna: int | None = None,
    n_rna_rna: int | None = None,
    n_dna_rna: int | None = None,
    n_rna_ligand: int | None = None,
    random_seed: int | None = None,
) -> None:
    """Subsamples a fixed number of interfaces per type.

    Follows AF3 SI 5.8 Multimer Selection Step 4. The function subsamples a specific
    number of interfaces per type, then returns a reduced cache only containing those
    interfaces.

    Args:
        dataset_cache (DatasetCache):
            The cache to subsample.
        n_protein_protein (int | None):
            The number of protein-protein interfaces to sample. Default is 600.
        n_protein_dna (int | None):
            The number of protein-DNA interfaces to sample. Default is 100.
        n_dna_dna (int | None):
            The number of DNA-DNA interfaces to sample. Default is 100.
        n_protein_ligand (int | None):
            The number of protein-ligand interfaces to sample. Default is 600.
        n_dna_ligand (int | None):
            The number of DNA-ligand interfaces to sample. Default is 50.
        n_ligand_ligand (int | None):
            The number of ligand-ligand interfaces to sample. Default is 200.
        n_protein_rna (int | None):
            The number of protein-RNA interfaces to sample. Default is None, which means
            that all protein-RNA interfaces will be selected.
        n_rna_rna (int | None):
            The number of RNA-RNA interfaces to sample. Default is None, which means
            that all RNA-RNA interfaces will be selected.
        n_dna_rna (int | None):
            The number of DNA-RNA interfaces to sample. Default is None, which means
            that all DNA-RNA interfaces will be selected.
        n_rna_ligand (int | None):
            The number of RNA-ligand interfaces to sample. Default is None, which means
            that all RNA-ligand interfaces will be selected.
        random_seed (int | None):
            The random seed to use for sampling. Default is None.

    Returns:
        None, the cache is updated in-place.
    """
    if random_seed is not None:
        random.seed(random_seed)

    interface_datapoints_by_type = {
        "protein_protein": [],
        "protein_dna": [],
        "dna_dna": [],
        "protein_ligand": [],
        "dna_ligand": [],
        "ligand_ligand": [],
        "protein_rna": [],
        "rna_rna": [],
        "dna_rna": [],
        "rna_ligand": [],
    }
    n_samples_by_type = {
        "protein_protein": n_protein_protein,
        "protein_dna": n_protein_dna,
        "dna_dna": n_dna_dna,
        "protein_ligand": n_protein_ligand,
        "dna_ligand": n_dna_ligand,
        "ligand_ligand": n_ligand_ligand,
        "protein_rna": n_protein_rna,
        "rna_rna": n_rna_rna,
        "dna_rna": n_dna_rna,
        "rna_ligand": n_rna_ligand,
    }

    for pdb_id, structure_data in dataset_cache.structure_data.items():
        for interface_id in structure_data.interfaces:
            chain_1, chain_2 = interface_id.split("_")
            chain_1_type = structure_data.chains[chain_1].molecule_type
            chain_2_type = structure_data.chains[chain_2].molecule_type

            molecule_types = (chain_1_type, chain_2_type)

            n_protein = molecule_types.count(MoleculeType.PROTEIN)
            n_dna = molecule_types.count(MoleculeType.DNA)
            n_rna = molecule_types.count(MoleculeType.RNA)
            n_ligand = molecule_types.count(MoleculeType.LIGAND)

            if n_protein == 2:
                interface_type = "protein_protein"
            elif n_protein == 1 and n_dna == 1:
                interface_type = "protein_dna"
            elif n_dna == 2:
                interface_type = "dna_dna"
            elif n_protein == 1 and n_ligand == 1:
                interface_type = "protein_ligand"
            elif n_dna == 1 and n_ligand == 1:
                interface_type = "dna_ligand"
            elif n_ligand == 2:
                interface_type = "ligand_ligand"
            elif n_protein == 1 and n_rna == 1:
                interface_type = "protein_rna"
            elif n_rna == 2:
                interface_type = "rna_rna"
            elif n_dna == 1 and n_rna == 1:
                interface_type = "dna_rna"
            elif n_rna == 1 and n_ligand == 1:
                interface_type = "rna_ligand"
            else:
                continue

            interface_datapoints_by_type[interface_type].append(
                InterfaceDataPoint(pdb_id, interface_id)
            )

    subsampled_interface_datapoints = []

    for interface_type, interface_datapoints in interface_datapoints_by_type.items():
        n_samples = n_samples_by_type[interface_type]

        # If None, include all samples
        if n_samples is None:
            subsampled_interface_datapoints.extend(interface_datapoints)
        else:
            subsampled_interface_datapoints.extend(
                random.sample(interface_datapoints, n_samples)
            )

    filter_cache_by_specified_interfaces(
        dataset_cache, set(subsampled_interface_datapoints)
    )


def check_interface_metric_eligibility(
    interface_id: str,
    interface_data: ValidationDatasetInterfaceData,
    chain_dict: dict[str, ValidationDatasetChainData],
    reference_mol_dict: ValidationDatasetReferenceMoleculeData,
    min_ranking_model_fit: float = 0.5,
) -> bool:
    """Decides whether an interface is eligible for validation metric inclusion.

    AF3 SI 5.8 is a bit ambiguous about which interfaces should be included in the final
    validation metrics for the multimer and monomer set. We decided to apply the
    interface criteria of low-homology and sufficient ligand-quality + single-residue
    ligand inclusion throughout both sets. Note that whether an interface is effectively
    included still depends on the subsampling logic in SI 5.8.

    Args:
        interface_id (str):
            The interface ID to check.
        interface_data (ValClusteredDatasetInterfaceData):
            The interface data to check.
        chain_dict (dict[str, ValClusteredDatasetChainData]):
            The dictionary of chain data for the structure the interface belongs to.
        reference_molecule_dict (ValClusteredDatasetReferenceMoleculeData):
            The dictionary of reference molecule data for the cache.
        min_ranking_model_fit (float):
            The minimum ranking model fit for ligands to be included. Default is 0.5.

    Returns:
        bool:
            Whether the interface is eligible for validation metric inclusion.
    """
    if not interface_data.low_homology:
        return False

    # Check if any interface chain is a ligand not meeting the criteria
    chain_1, chain_2 = interface_id.split("_")

    for chain_id in (chain_1, chain_2):
        chain_data = chain_dict[chain_id]

        if chain_data.molecule_type == MoleculeType.LIGAND:
            # Check that fit is above threshold
            if chain_data.ranking_model_fit < min_ranking_model_fit:
                return False

            # Check that ligand is single-residue
            mol_id = chain_data.reference_mol_id
            if reference_mol_dict[mol_id].residue_count > 1:
                return False

    return True


def assign_interface_metric_eligibility_labels(
    val_dataset_cache: ValidationDatasetCache,
    min_ranking_model_fit: float = 0.5,
) -> None:
    """Sets the metric_eligible attribute for all interfaces in the cache.

    This function will set the metric_eligible attribute for all interfaces in the
    cache. Following SI 5.8, we define interface metric eligibility as:

    - The interface has low-homology to the training set
    - If the interface contains ligands, then all ligands have a residue count of 1 and
      a ranking model fit above a certain threshold

    While SI 5.8 is ambiguous about this, we effectively apply those criteria not only
    to the multimer set but also any ligand-containing interface in the monomer set.

    Args:
        val_dataset_cache (ValClusteredDatasetCache):
            The cache to assign metric eligibility labels to.
        min_ranking_model_fit (float):
            The minimum ranking model fit for ligands to be included. Default is 0.5.

    Returns:
        None, the cache is updated in-place.
    """
    for structure_data in val_dataset_cache.structure_data.values():
        for interface_id, interface_data in structure_data.interfaces.items():
            interface_data.metric_eligible = check_interface_metric_eligibility(
                interface_id=interface_id,
                interface_data=interface_data,
                chain_dict=structure_data.chains,
                reference_mol_dict=val_dataset_cache.reference_molecule_data,
                min_ranking_model_fit=min_ranking_model_fit,
            )


class ValidationSummaryStats(NamedTuple):
    """Summary statistics about chains/interfaces for a validation set cache.

    Attributes:
        n_pdb_ids (int):
            The number of PDB IDs in the cache.
        n_chains (int):
            The total number of chains in the cache.
        n_low_homology_chains (int):
            The number of chains with low homology.
        n_scored_chains (int):
            The number of chains that will be scored with validation metrics.
        n_interfaces (int):
            The total number of interfaces in the cache.
        n_low_homology_interfaces (int):
            The number of interfaces with low homology.
        n_scored_interfaces (int):
            The number of interfaces that will be scored with validation metrics.
    """

    n_pdb_ids: int
    n_chains: int
    n_low_homology_chains: int
    n_scored_chains: int
    n_interfaces: int
    n_low_homology_interfaces: int
    n_scored_interfaces: int


def get_validation_summary_stats(
    structure_data: dict[str, ValidationDatasetStructureData],
) -> ValidationSummaryStats:
    """Gets summary statistics for a validation dataset cache.

    Args:
        structure_data: dict[str, ValClusteredDatasetStructureData]
            The structure data to log statistics for.

    Returns:
        A NamedTuple with the following fields:

            - n_pdb_ids (int)
            - n_chains (int)
            - n_low_homology_chains (int)
            - n_scored_chains (int)
            - n_interfaces (int)
            - n_low_homology_interfaces (int)
            - n_scored_interfaces (int)
    """
    n_pdb_ids = len(structure_data)

    n_chains = 0
    n_low_homology_chains = 0
    n_scored_chains = 0
    n_interfaces = 0
    n_low_homology_interfaces = 0
    n_scored_interfaces = 0

    for structure_data_entry in structure_data.values():
        for chain_data in structure_data_entry.chains.values():
            n_chains += 1
            if chain_data.low_homology:
                n_low_homology_chains += 1
            if chain_data.use_metrics:
                n_scored_chains += 1

        for interface_data in structure_data_entry.interfaces.values():
            n_interfaces += 1
            if interface_data.low_homology:
                n_low_homology_interfaces += 1
            if interface_data.use_metrics:
                n_scored_interfaces += 1

    return ValidationSummaryStats(
        n_pdb_ids=n_pdb_ids,
        n_chains=n_chains,
        n_low_homology_chains=n_low_homology_chains,
        n_scored_chains=n_scored_chains,
        n_interfaces=n_interfaces,
        n_low_homology_interfaces=n_low_homology_interfaces,
        n_scored_interfaces=n_scored_interfaces,
    )


def add_ligand_data_to_monomer_cache(
    unfiltered_structure_data: ValidationDatasetCache,
    monomer_structure_data: dict[str, ValidationDatasetCache],
) -> None:
    """Expands the validation monomer set with valid ligand chains and interfaces.

    Following AF3 SI 5.8, this function will add back ligand chains and interfaces to
    the monomer set. Ligand chains are included if the ligand is low-homology, and
    ligand interfaces are included if they are low-homology and the ligand is
    single-residue and has a ranking model fit above a certain threshold.

    Args:
        unfiltered_cache: ValClusteredDatasetCache
            Structure cache that contains full information of the monomer targets before
            any chain subsetting.
        monomer_structure_data: dict[str, ValClusteredDatasetStructureData]
            The selected and subsampled low-homology polymer chains of the monomer set
            after the token-count filtering in SI 5.8 Step 5.

    Returns:
        None, the monomer_structure_data is updated in-place.
    """
    monomer_pdb_ids = set(monomer_structure_data.keys())

    for pdb_id in monomer_pdb_ids:
        target_chain_data = unfiltered_structure_data[pdb_id].chains
        target_interface_data = unfiltered_structure_data[pdb_id].interfaces

        # TODO: remove
        assert (
            sum(
                chain.molecule_type
                in [MoleculeType.PROTEIN, MoleculeType.DNA, MoleculeType.RNA]
                for chain in target_chain_data.values()
            )
            == 1
        )

        # Add ligand chains
        for chain_id, chain_data in target_chain_data.items():
            if (
                chain_data.molecule_type == MoleculeType.LIGAND
                and chain_data.low_homology
            ):
                monomer_structure_data[pdb_id].chains[chain_id] = deepcopy(chain_data)

        # Add ligand interfaces
        for interface_id, interface_data in target_interface_data.items():
            chain_1, chain_2 = interface_id.split("_")
            chain_1_data = target_chain_data[chain_1]
            chain_2_data = target_chain_data[chain_2]

            assert (
                chain_1_data.molecule_type == MoleculeType.LIGAND
                or chain_2_data.molecule_type == MoleculeType.LIGAND
            )

            if interface_data.metric_eligible:
                monomer_structure_data[pdb_id].interfaces[interface_id] = deepcopy(
                    interface_data
                )


def select_final_validation_data(
    unfiltered_cache: ValidationDatasetCache,
    monomer_structure_data: dict[str, ValidationDatasetStructureData],
    multimer_structure_data: dict[str, ValidationDatasetStructureData],
) -> None:
    """Selects the final targets and marks chains/interfaces to score on.

    This will create the final validation dataset cache by subsetting the unfiltered
    cache only to the relevant PDB-IDs, and then turning on the use_metrics flag only
    for select chains and interfaces coming out of the multimer and monomer sets. Note
    that we are not including all low-homology chains and interfaces of each target in
    the final validation set, but only those that are part of the selected monomer and
    multimer sets.

    Args:
        unfiltered_cache: ValClusteredDatasetCache
            Preliminary validation dataset cache corresponding to the full proto
            validation set, after the initial time- and token-based filtering.
        monomer_structure_data: dict[str, ValClusteredDatasetStructureData]
            The monomer set of SI 5.8, containing the subsampled low-homology polymer
            chains and metric-eligible low-homology interfaces.
        multimer_structure_data: dict[str, ValClusteredDatasetStructureData]
            The multimer set of SI 5.8, containing subsampled low-homology interfaces
            and their constituent chains.


    Returns:
        None, the filtered_structure_data is updated in-place.
    """
    # First subset the unfiltered cache to only the relevant PDB-IDs
    relevant_pdb_ids = set(monomer_structure_data.keys()) | set(
        multimer_structure_data.keys()
    )
    structure_data = unfiltered_cache.structure_data
    structure_data = {pdb_id: structure_data[pdb_id] for pdb_id in relevant_pdb_ids}

    for pdb_id, structure_data_entry in structure_data.items():
        # All monomer chains/interfaces are already valid for metric inclusion
        if pdb_id in monomer_structure_data:
            for chain_id in monomer_structure_data[pdb_id].chains:
                structure_data_entry.chains[chain_id].use_metrics = True

            for interface_id in monomer_structure_data[pdb_id].interfaces:
                structure_data_entry.interfaces[interface_id].use_metrics = True

        # Multimer interfaces are all valid for metric inclusion but chains may not be
        # (because low-homology for interfaces is based on co-occurrence of two chains
        # so they could have high-homology individually)
        if pdb_id in multimer_structure_data:
            for interface_id in multimer_structure_data[pdb_id].interfaces:
                structure_data_entry.interfaces[interface_id].use_metrics = True

            for chain_id in multimer_structure_data[pdb_id].chains:
                # Extra low-homology check
                if multimer_structure_data[pdb_id].chains[chain_id].low_homology:
                    structure_data_entry.chains[chain_id].use_metrics = True

    unfiltered_cache.structure_data = structure_data
