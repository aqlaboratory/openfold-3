"""This module contains building blocks for MSA processing."""

import dataclasses
import logging
import math
from typing import Sequence, Union

import numpy as np
import pandas as pd

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=False)
class MsaArray:
    """Class representing a parsed MSA file.

    The metadata attribute gets updated in certain functions of the MSA preparation.

    Attributes:
        msa (np.array):
            A 2D numpy array containing the aligned sequences.
        deletion_matrix (np.array):
            A 2D numpy array containing the cumulative deletion counts up to each
            position for each row in the MSA.
        metadata (pd.DataFrame):
            A list of metadata parsed from sequence headers of the MSA. If not provided,
            an empty DataFrame is assigned."""

    msa: np.ndarray[str]
    deletion_matrix: np.ndarray[int]
    metadata: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)

    def __len__(self):
        return self.msa.shape[0]

    def truncate(self, max_seq_count: int) -> None:
        """Truncate the MSA to a maximum number of sequences.

        Args:
            max_seq_count (int): Number of sequences to keep in the MSA.

        Returns:
            None
        """

        if not isinstance(max_seq_count, int) | (max_seq_count == math.inf):
            raise ValueError("max_seq_count should be an integer or math.inf.")

        if self.__len__() > max_seq_count:
            if max_seq_count == math.inf:
                max_seq_count = self.__len__()

            self.msa = self.msa[:max_seq_count, :]
            self.deletion_matrix = self.deletion_matrix[:max_seq_count, :]
            self.metadata = (
                self.metadata[:max_seq_count]
                if isinstance(self.metadata, list)
                else self.metadata.iloc[: (max_seq_count - 1)]
            )


@dataclasses.dataclass(frozen=False)
class MsaArrayCollection:
    """Class representing a collection MSAs for a single sample.

    This class can be in one of two states:
        1.) parsed state:
            attributes rep_id_to_msa and rep_id_to_query_seq populated after parsing and
            chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa
            unpopulated.
        2.) processed state:
            attributes rep_id_to_msa and rep_id_to_query_seq unpopulated and
            chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa
            populated after processing.

    Attributes chain_id_to_rep_id, chain_id_to_mol_type, and num_cols are populated in
    both states.

    Attributes:
        rep_id_to_msa (dict[str, dict[str, MsaArray]]):
            Dictionary mapping representative chain IDs to dictionaries of Msa objects.
        rep_id_to_query_seq (dict[str, np.ndarray[str]]):
            Dictionary mapping representative chain IDs to numpy arrays of their
            corresponding query sequences.
        chain_id_to_query_seq (dict[str, MsaArray]):
            Dictionary mapping chain IDs to Msa objects containing query sequence data.
        chain_id_to_paired_msa (dict[str, MsaArray]):
            Dictionary mapping chain IDs to Msa objects containing paired MSA data.
        chain_id_to_main_msa (dict[str, MsaArray]):
            Dictionary mapping chain IDs to Msa objects containing main MSA data.
        chain_id_to_rep_id (dict[str, str]):
            Dictionary mapping chain IDs to representative chain IDs.
        chain_id_to_mol_type (dict[str, str]):
            Dictionary mapping chain IDs to the molecule type.
        num_cols (dict[str, int]):
            Dict mapping representative chain ID to the number of columns in the MSA.
    """

    # Core attributes
    chain_id_to_rep_id: dict[str, str]
    chain_id_to_mol_type: dict[str, str]
    num_cols: dict[str, int]  # maybe not needed?

    # State parsed attributes
    rep_id_to_msa: dict[str, dict[str, MsaArray]] = dataclasses.field(
        default_factory=dict
    )
    rep_id_to_query_seq: dict[str, np.ndarray[str]] = dataclasses.field(
        default_factory=dict
    )

    # State processed attributes
    chain_id_to_query_seq: dict[str, MsaArray] = dataclasses.field(default_factory=dict)
    chain_id_to_paired_msa: dict[str, MsaArray] = dataclasses.field(
        default_factory=dict
    )
    chain_id_to_main_msa: dict[str, MsaArray] = dataclasses.field(default_factory=dict)

    def set_state_parsed(self, rep_id_to_msa, rep_id_to_query_seq):
        """Set the state to parsed."""
        self.rep_id_to_msa = rep_id_to_msa
        self.rep_id_to_query_seq = rep_id_to_query_seq
        self.chain_id_to_query_seq = {}
        self.chain_id_to_paired_msa = {}
        self.chain_id_to_main_msa = {}

    def set_state_processed(
        self, chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa
    ):
        """Set the state to processed."""
        self.chain_id_to_query_seq = chain_id_to_query_seq
        self.chain_id_to_paired_msa = chain_id_to_paired_msa
        self.chain_id_to_main_msa = chain_id_to_main_msa
        self.rep_id_to_msa = {}
        self.rep_id_to_query_seq = {}


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-homo-mono")
def find_monomer_homomer(msa_array_collection: MsaArrayCollection) -> bool:
    """Determines if the sample is a monomer or homomer.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        bool: Whether the sample is a monomer or a full homomer.
    """
    # Extract chain IDs and representative chain IDs
    chain_id_to_rep_id = {
        chain_id: rep_id
        for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items()
        if msa_array_collection.chain_id_to_mol_type[chain_id] == "PROTEIN"
    }
    chain_ids, representative_chain_ids = (
        list(chain_id_to_rep_id.keys()),
        list(set(chain_id_to_rep_id.values())),
    )

    return (len(chain_ids) == 1) | (
        (len(representative_chain_ids) == 1) & (len(chain_ids) > 1)
    )


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-query")
def create_query_seqs(msa_collection: MsaArrayCollection) -> dict[int, MsaArray]:
    """Extracts and expands the query sequences and deletion matrices.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[int, Msa]:
            Dict of Msa objects containing the query sequence and deletion matrix
            for each chain, indexed by chain id.
    """
    return {
        k: MsaArray(
            msa=msa_collection.rep_id_to_query_seq[v],
            deletion_matrix=np.zeros(
                msa_collection.rep_id_to_query_seq[v].shape, dtype=int
            ),
            metadata=pd.DataFrame(),
        )
        for (k, v) in msa_collection.chain_id_to_rep_id.items()
    }


def extract_uniprot_hits(
    msa_array_collection: MsaArrayCollection,
) -> dict[str, MsaArray]:
    """Parses out UniProt Msa objects for unique protein chains from the MsaCollection.

    This function does not return UniProt MSAs for chains that only contain the query
    i.e. single sequence MSAs.

    Args:
        msa_array_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.
    """
    protein_rep_ids = set(
        rep_id
        for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items()
        if msa_array_collection.chain_id_to_mol_type[chain_id] == "PROTEIN"
    )
    rep_ids = msa_array_collection.rep_id_to_msa.keys()

    # Get uniprot hits, exclude MSAs only with query
    uniprot_hits = {}
    for rep_id in rep_ids:
        if rep_id not in protein_rep_ids:
            continue

        rep_msa_map_per_chain = msa_array_collection.rep_id_to_msa[rep_id]
        uniprot_msa = (
            rep_msa_map_per_chain.get("uniprot_hits")
            if "uniprot_hits" in rep_msa_map_per_chain
            else rep_msa_map_per_chain.get("uniprot")
        )
        if uniprot_msa is not None and len(uniprot_msa) > 1:
            uniprot_hits[rep_id] = uniprot_msa

    return uniprot_hits


def process_uniprot_metadata(msa: MsaArray) -> None:
    """Reformats the metadata of an Msa object parsed from a UniProt MSA.

    This function expects
    1) the 1st row to contain the query header - it is excluded from the
    metadata DataFrame.
    2) the rest of the rows to be of the format
        tr|<UniProt ID>|<UniProt ID>_<species ID>/<start idx>-end idx;
    for example:
        tr|A0A1W9RZR3|A0A1W9RZR3_9BACT/19-121

    The list of headers are converted into a DataFrame containing the uniprot_id,
    species_id, chain_start and chain_end columns. If the Msa only contains the
    query sequence, an empty DataFrame is assigned to the metadata attribute.

    Args:
        msa (Msa): parsed Msa object

    Returns:
        None
    """

    # Embed into DataFrame
    if len(msa.metadata) == 1:
        # Empty DataFrame for UniProt MSAs that only contain the query sequence
        metadata = pd.DataFrame({"raw": msa.metadata})
        metadata = pd.DataFrame()
    else:
        metadata = pd.DataFrame({"raw": msa.metadata[1:]})
        metadata = metadata["raw"].str.split(r"[|_/:-]", expand=True)
        metadata.columns = [
            "tr",
            "uniprot_id",
            "uniprot_id_copy",
            "species_id",
            "chain_start",
            "chain_end",
        ]
        metadata = metadata[["uniprot_id", "species_id"]]

    msa.metadata = metadata


def sort_msa_by_distance_to_query(msa: MsaArray) -> None:
    """Reorders the MSA array based on the distance to the query sequence.

    Reorders all class attributes of the MSA object.

    Args:
        msa (Msa): parsed Msa object

    Returns:
        None
    """
    msa_array = msa.msa
    distance_to_query = np.sum(msa_array == msa_array[0, :], axis=-1) / float(
        sum(msa_array[0, :] != "-")
    )
    sorting_indices = np.argsort(distance_to_query)[::-1]
    msa.msa = msa_array[sorting_indices, :]
    msa.metadata = msa.metadata.iloc[sorting_indices[1:] - 1]
    msa.deletion_matrix = msa.deletion_matrix[sorting_indices, :]


def count_species_per_chain(
    uniprot_hits: dict[str, MsaArray],
) -> tuple[np.ndarray[np.int32], list[str]]:
    """Counts the occurrences of sequences from species in each chain's UniProt MSA.

    Args:
        uniprot_hits (dict[str, Msa]):
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.

    Returns:
        np.ndarray[np.int32], list[str]:
            The array of occurrence counts (number of chains x number of unique species)
            and the list species with at least one sequence among MSAs of all chains.
    """
    species = []
    _ = [
        species.extend(set(uniprot_hits[chain_id].metadata["species_id"]))
        for chain_id in uniprot_hits
    ]
    species = list(set(species))
    species_index = np.arange(len(species))
    species_index_map = {species[i]: i for i in species_index}

    # Get lists of species per chain
    species_index_per_chain = [
        np.array(
            uniprot_hits[chain_id]
            .metadata["species_id"]
            .apply(lambda x: species_index_map[x])
        )
        for chain_id in uniprot_hits
    ]

    # Combine all lists into one array
    all_lists = np.concatenate(species_index_per_chain)

    # List to keep track of which list each element came from
    list_indices = np.concatenate(
        [np.array([i] * len(lst)) for i, lst in enumerate(species_index_per_chain)]
    )

    # Get unique integers and their inverse mapping to reconstruct original array
    unique_integers, inverse_indices = np.unique(all_lists, return_inverse=True)

    # Initialize the 2D array to count occurrences
    count_array = np.zeros((len(uniprot_hits), len(unique_integers)), dtype=int)

    # Use np.add.at for unbuffered in-place addition
    np.add.at(count_array, (list_indices, inverse_indices), 1)

    return count_array, species


def get_pairing_masks(
    count_array: np.ndarray[np.int32], mask_keys: Sequence[str]
) -> np.ndarray[np.bool_]:
    """Generates masks for the pairing process.

    Useful for excluding things like species that occur only in one chain (will not be
    pairable) or species that occur too frequently in the MSA of a single chain (as done
    in the AF2-Multimer pairing code:
    https://github.com/google-deepmind/alphafold/blob/main/alphafold/data/msa_pairing.py#L216).

    Args:
        count_array (np.ndarray[np.int32]):
            The array of species occurrence counts per chain.
        mask_keys (Sequence[str]):
            List of strings indicating which mask to add.

    Returns:
        np.ndarray[np.bool_]: The union of all masks to apply during pairing.
    """
    pairing_masks = np.ones(count_array.shape[1], dtype=bool)

    if "shared_by_two" in mask_keys:
        # Find species that are shared by at least two chains
        pairing_masks = pairing_masks & (np.sum(count_array != 0, axis=0) > 1)

    if "less_than_600" in mask_keys:
        # Find species that occur more than 600 times in any single chain
        pairing_masks = pairing_masks & (
            np.sum(count_array <= 600, axis=0) == count_array.shape[0]
        )

    return pairing_masks


def find_pairing_indices(
    count_array: np.ndarray[np.int32],
    pairing_masks: np.ndarray[np.bool_],
    max_rows_paired: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The main function for finding indices that pair rows in the MSA arrays.

    This function follows the AF2-Multimer strategy for pairing rows of UniProt MSAs but
    excludes block-diagonal elements (unpaired rows).

    Args:
        count_array (np.ndarray[np.int32]):
            The array of species occurrence counts per chain
        pairing_masks (np.ndarray[np.bool_]):
            The union of all masks to apply during pairing.
        max_rows_paired (int):
            The maximum number of rows to pair.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple of arrays containing the indices that pair rows in MSAs of an
            assembly across chains and the mask for partially paired rows for each chain
            that cannot be fully paired.
    """
    # Apply filters
    species_index = np.arange(count_array.shape[-1])
    count_array_filtered = count_array[:, pairing_masks]
    species_index_filtered = species_index[pairing_masks]
    is_in_chain_per_species = count_array != 0

    # Iterative row-subtraction
    n_unique_chains = count_array.shape[0]
    paired_species_rows = []  # species indices in the MSA feature format
    missing_species_rows = []  # mask for missing species per chain
    n_rows = 0  # number of paired rows
    for n in np.arange(1, n_unique_chains + 1)[::-1]:
        # Break if n is 1 since we are excluding block-diagonal elements
        if n == 1:
            break

        # Find which species are shared by exactly n chains
        is_in_n_chains = sum(count_array_filtered != 0) == n

        # Find the lowest number of sequences across chains from shared species
        min_in_n_chains = np.min(count_array_filtered[:, is_in_n_chains], axis=0)

        # Subset species indices to those shared by n chains
        species_in_n_chains = species_index_filtered[is_in_n_chains]

        # Expand filtered species index across occurrences and chains
        cols = np.tile(
            np.repeat(species_in_n_chains, min_in_n_chains), (n_unique_chains, 1)
        )
        n_rows += cols.shape[1]
        paired_species_rows.append(cols.T)

        # Create mask for elements where species is missing from chain
        missing_species_mask = is_in_chain_per_species[:, cols[0, :]]
        missing_species_rows.append(missing_species_mask.T)

        # Subtract min per row for shared species
        count_array_filtered[:, is_in_n_chains] -= min_in_n_chains

        # If row cutoff reached, crop final arrays to the row cutoff and break
        if n_rows >= max_rows_paired:
            n_rows_final = max_rows_paired - sum(
                [rows.shape[0] for rows in paired_species_rows[:-1]]
            )
            paired_species_rows[-1] = paired_species_rows[-1][: n_rows_final + 1, :]
            missing_species_rows[-1] = missing_species_rows[-1][: n_rows_final + 1, :]
            break

    # Concatenate all paired arrays into a single paired array
    paired_rows_index = np.concatenate(paired_species_rows, axis=0)
    missing_rows_index = np.concatenate(missing_species_rows, axis=0)

    return paired_rows_index, missing_rows_index


def map_to_paired_msa_per_chain(
    msa_array_collection: MsaArrayCollection,
    uniprot_hits: dict[str, MsaArray],
    paired_rows_index: np.ndarray[int],
    missing_rows_index: np.ndarray[int],
    species: list,
) -> dict[str, MsaArray]:
    """Maps paired species indices to MSA row indices.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        uniprot_hits (dict[str, Msa]):
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.
        paired_rows_index (np.ndarray[np.int32]):
            Array containing the indices that pair rows in MSAs of an assembly across
            chains.
        missing_rows_index (np.ndarray[np.int32]):
            Mask for partially paired rows for each chain that cannot be fully paired.
        species (list):
            List of species with at least one sequence among MSAs of all chains, in the
            order used by entries ofe th paired_rows_index.

    Returns:
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing the paired MSAs and paired
            deletion matrices. Metadata fields are empty.
    """

    # Map species indices back to MSA row indices
    # Pre-allocate MSA objects, including those without UniProt hits
    paired_msa_per_chain = {
        rep_id: MsaArray(
            msa=np.full((paired_rows_index.shape[0], seq.shape[-1]), "-"),
            deletion_matrix=np.zeros(
                (paired_rows_index.shape[0], seq.shape[-1]),
                dtype=int,
            ),
            metadata=pd.DataFrame(),
        )
        for rep_id, seq in msa_array_collection.rep_id_to_query_seq.items()
    }

    # For each chain, sort MSA rows by the paired species indices
    species_index = {v: i for i, v in enumerate(species)}
    for chain_idx, chain_id in enumerate(uniprot_hits):
        # Get the array of species for each aligned sequences to the query chain
        species_array = np.array(
            uniprot_hits[chain_id].metadata["species_id"].to_numpy()
        )

        # Map to numerical using the species index
        row_to_species = np.vectorize(species_index.get)(species_array)

        # Get paired species ids for chain
        paired_row_index_of_chain = paired_rows_index[:, chain_idx]

        # For each paired species row for the chain, find the position of
        # msa rows with the same species
        species_matches = row_to_species == paired_row_index_of_chain[:, np.newaxis]

        # Find MSA row indices for each paired species row
        used_rows = set()
        msa_rows = np.zeros(paired_row_index_of_chain.shape[0], dtype=int)
        for row_idx, row in enumerate(species_matches):
            species_matches_row = np.where(row)[0]
            first_true_row = species_matches_row[
                np.argmax(~np.isin(species_matches_row, list(used_rows)))
            ]
            msa_rows[row_idx] = first_true_row
            used_rows.add(first_true_row)

        # Update MSA and deletion matrix with paired data
        paired_msa_per_chain[chain_id].msa[missing_rows_index[:, chain_idx], :] = (
            uniprot_hits[chain_id].msa[msa_rows, :]
        )
        paired_msa_per_chain[chain_id].deletion_matrix[
            missing_rows_index[:, chain_idx], :
        ] = uniprot_hits[chain_id].deletion_matrix[msa_rows, :]

    return paired_msa_per_chain


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-paired")
def create_paired(
    msa_array_collection: MsaArrayCollection, max_rows_paired: int
) -> tuple[dict[str, MsaArray], dict[str, MsaArray]]:
    """Creates paired MSA arrays from UniProt MSAs.

    Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on species
    IDs, but excludes the block-diagonal rows, i.e. only includes rows which can be at
    least partially paired, as suggested by the AF3 SI.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        paired_row_cutoff (int):
            The maximum number of rows to pair.

    Returns:
        dict[str, Msa]:
            Paired MSAs and deletion matrices for each chain.
    """
    # Get parsed uniprot hits
    uniprot_hits = extract_uniprot_hits(msa_array_collection)

    # Ensure there are at least two chains with UniProt hits after filtering
    if len(uniprot_hits) <= 1:
        return None, None

    # Process uniprot headers and calculate distance to query
    _ = [process_uniprot_metadata(uniprot_hits[chain_id]) for chain_id in uniprot_hits]
    _ = [
        sort_msa_by_distance_to_query(uniprot_hits[chain_id])
        for chain_id in uniprot_hits
    ]

    # Count species occurrences per chain
    count_array, species = count_species_per_chain(uniprot_hits)

    # Get pairing masks
    pairing_masks = get_pairing_masks(count_array, ["shared_by_two", "less_than_600"])

    # No valid pairs, skip MSA pairing
    if not np.any(pairing_masks):
        return None, None

    # Find species indices that pair rows
    paired_rows_index, missing_rows_index = find_pairing_indices(
        count_array,
        pairing_masks,
        max_rows_paired,
    )

    # Map species indices back to MSA row indices
    paired_msa_per_chain = map_to_paired_msa_per_chain(
        msa_array_collection,
        uniprot_hits,
        paired_rows_index,
        missing_rows_index,
        species,
    )

    # Expand paired MSAs across all chains
    paired_msas = {}
    for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items():
        rep_paired_msa = paired_msa_per_chain[rep_id]
        paired_msas[chain_id] = MsaArray(
            msa=rep_paired_msa.msa,
            deletion_matrix=rep_paired_msa.deletion_matrix,
            metadata=pd.DataFrame(),
        )

    return paired_msa_per_chain, paired_msas


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-main")
def create_main(
    msa_array_collection: MsaArrayCollection,
    paired_msa_per_chain: Union[dict[str, MsaArray], None],
    aln_order: list[str],
) -> dict[str, MsaArray]:
    """Creates main MSA arrays from non-UniProt MSAs.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        paired_msa_per_chain (Union[dict[str, Msa], None]):
            Dict of paired Msa objects per chain.
        aln_order (list[str]):
            The order in which to concatenate the MSA arrays vertically.

    Returns:
        dict[str, Msa]:
            List of Msa objects containing the main MSA arrays and deletion matrices
            for each chain.
    """
    # Iterate over representatives
    rep_main_msas = {}
    for rep_id, chain_data in msa_array_collection.rep_id_to_msa.items():
        chain_data = msa_array_collection.rep_id_to_msa[rep_id]

        # Get MSAs forming the main MSA and deletion matrices from all non-UniProt MSAs
        main_msa_redundant = np.concatenate(
            [chain_data[aln].msa for aln in aln_order if aln in chain_data],
            axis=0,
        )
        main_deletion_matrix_redundant = np.concatenate(
            [chain_data[aln].deletion_matrix for aln in aln_order if aln in chain_data],
            axis=0,
        )

        # Get paired MSAs if any and deduplicate
        if paired_msa_per_chain is not None:
            # Get set of paired rows and find unpaired rows not in this set
            paired_row_set = {tuple(i) for i in paired_msa_per_chain[rep_id].msa}
            is_unique = ~np.array(
                [tuple(row) in paired_row_set for row in main_msa_redundant]
            )
        else:
            is_unique = np.ones(main_msa_redundant.shape[0], dtype=bool)

        rep_main_msas[rep_id] = MsaArray(
            msa=main_msa_redundant[is_unique, :],
            deletion_matrix=main_deletion_matrix_redundant[is_unique, :],
            metadata=pd.DataFrame(),
        )

    # Reindex dict from representatives to chain IDs
    main_msas = {}
    for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items():
        main_msas[chain_id] = rep_main_msas[rep_id]

    return main_msas
