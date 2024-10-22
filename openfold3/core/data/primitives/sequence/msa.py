"""This module contains building blocks for MSA processing."""

import dataclasses
import logging
import math
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.resources.residues import (
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_POS,
    MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1,
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=False)
class MsaParsed:
    """Class representing a parsed MSA file.

    The metadata attribute gets updated in certain functions of the MSA preparation.

    Attributes:
        msa (np.array):
            A 2D numpy array containing the aligned sequences.
        deletion_matrix (np.array):
            A 2D numpy array containing the cumulative deletion counts up to each
            position for each row in the MSA.
        metadata (Optional[Sequence[str]]):
            A list of metadata parsed from sequence headers of the MSA."""

    msa: np.array
    deletion_matrix: np.array
    metadata: Optional[Sequence[str]]

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
class MsaCollection:
    """Class representing a collection MSAs for a single sample.

    Attributes:
        rep_msa_map (dict[str, dict[str, Msa]]):
            Dictionary mapping representative chain IDs to dictionaries of Msa objects.
        rep_seq_map (dict[str, np.ndarray[np.str_]]):
            Dictionary mapping representative chain IDs to numpy arrays of their
            corresponding query sequences.
        chain_rep_map (dict[str, str]):
            Dictionary mapping chain IDs to representative chain IDs.
        chain_to_molecule_type (dict[str, str]):
            Dictionary mapping chain IDs to the molecule type.
        num_cols (dict[str, int]):
            Dict mapping representative chain ID to the number of columns in the MSA.
    """

    rep_msa_map: dict[str, dict[str, MsaParsed]]
    rep_seq_map: dict[str, np.ndarray[np.str_]]
    chain_rep_map: dict[str, str]
    chain_to_molecule_type: dict[str, str]
    num_cols: dict[str, int]


@dataclasses.dataclass(frozen=False)
class MsaProcessedCollection:
    """Class storing processed Msa data per chain.

    The stored MSAs are expanded across all redundant chains.

    Attributes:
        query_sequences (dict[int, Msa]):
            Dictionary mapping chain IDs to Msa objects containing query sequence data.
        main_msas (dict[int, Msa]):
            Dictionary mapping chain IDs to Msa objects containing main MSA data.
        paired_msas (Optional[dict[int, Msa]]):
            Dictionary mapping chain IDs to Msa objects containing paired MSA data."""

    query_sequences: dict[int, MsaParsed]
    main_msas: dict[int, MsaParsed]
    paired_msas: Optional[dict[int, MsaParsed]] = None


@dataclasses.dataclass(frozen=False)
class MsaSlice:
    """Class storing crop-to-alignment maps.

    Attributes:
        chain_rep_map (dict[int, str]):
            Dictionary mapping chain IDs to representative IDs used for finding the
            directory containing the alignments for the corresponding chains.
        tokens_in_chain (dict[int, dict[int, int]]):
            Dictionary mapping tokens that fall into the crop to corresponding residue
            indices in the matching alignment.
        chain_to_molecule_type (dict[str, str]):
            Dictionary mapping chain IDs to the molecule type.
    """

    chain_rep_map: dict[int, str]
    tokens_in_chain: dict[int, dict[int, int]]
    chain_to_molecule_type: dict[str, str]


@dataclasses.dataclass(frozen=False)
class MsaFeaturePrecursorAF3(MsaParsed):
    """Class representing the fully processed MSA arrays of an assembly.

    Subclass of MsaParsed.

    Args:
        MsaParsed (Type[MsaParsed]):
            Parsed MSA parent class.
    Attributes:
        n_rows_paired (int):
            Number of paired rows in the MSA array
        n_rows_main_per_chain (dict[int, int]):
            Dict mapping chain id to number of main MSA rows.
    """

    n_rows_paired: int
    msa_mask: np.ndarray
    msa_profile: np.ndarray
    deletion_mean: np.ndarray


def find_monomer_homomer(msa_collection: MsaCollection) -> bool:
    """Determines if the sample is a monomer or homomer.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        bool: Whether the sample is a monomer or a full homomer.
    """
    # Extract chain IDs and representative chain IDs
    chain_rep_map = {
        chain_id: rep_id
        for chain_id, rep_id in msa_collection.chain_rep_map.items()
        if msa_collection.chain_to_molecule_type[chain_id] == "PROTEIN"
    }
    chain_ids, representative_chain_ids = (
        list(chain_rep_map.keys()),
        list(set(chain_rep_map.values())),
    )

    return (len(chain_ids) == 1) | (
        (len(representative_chain_ids) == 1) & (len(chain_ids) > 1)
    )


def create_query_seqs(msa_collection: MsaCollection) -> dict[int, MsaParsed]:
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
        k: MsaParsed(
            msa=msa_collection.rep_seq_map[v],
            deletion_matrix=np.zeros(msa_collection.rep_seq_map[v].shape, dtype=int),
            metadata=pd.DataFrame(),
        )
        for (k, v) in msa_collection.chain_rep_map.items()
    }


def extract_uniprot_hits(msa_collection: MsaCollection) -> dict[str, MsaParsed]:
    """Parses out UniProt Msa objects for unique protein chains from the MsaCollection.

    This function does not return UniProt MSAs for chains that only contain the query
    i.e. single sequence MSAs.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.
    """
    protein_rep_ids = set(
        rep_id
        for chain_id, rep_id in msa_collection.chain_rep_map.items()
        if msa_collection.chain_to_molecule_type[chain_id] == "PROTEIN"
    )
    rep_ids = msa_collection.rep_msa_map.keys()

    # Get uniprot hits, exclude MSAs only with query
    uniprot_hits = {}
    for rep_id in rep_ids:
        if rep_id not in protein_rep_ids:
            continue

        rep_msa_map_per_chain = msa_collection.rep_msa_map[rep_id]
        uniprot_msa = (
            rep_msa_map_per_chain.get("uniprot_hits")
            if "uniprot_hits" in rep_msa_map_per_chain
            else rep_msa_map_per_chain.get("uniprot")
        )
        if uniprot_msa is not None and len(uniprot_msa) > 1:
            uniprot_hits[rep_id] = uniprot_msa

    return uniprot_hits


def process_uniprot_metadata(msa: MsaParsed) -> None:
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


def sort_msa_by_distance_to_query(msa: MsaParsed) -> None:
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
    uniprot_hits: dict[str, MsaParsed],
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
    paired_row_cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The main function for finding indices that pair rows in the MSA arrays.

    This function follows the AF2-Multimer strategy for pairing rows of UniProt MSAs but
    excludes block-diagonal elements (unpaired rows).

    Args:
        count_array (np.ndarray[np.int32]):
            The array of species occurrence counts per chain
        pairing_masks (np.ndarray[np.bool_]):
            The union of all masks to apply during pairing.
        paired_row_cutoff (int):
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
        if n_rows >= paired_row_cutoff:
            n_rows_final = paired_row_cutoff - sum(
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
    msa_collection: MsaCollection,
    uniprot_hits: dict[str, MsaParsed],
    paired_rows_index: np.ndarray[np.int32],
    missing_rows_index: np.ndarray[np.int32],
    species: list,
) -> dict[str, MsaParsed]:
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
        rep_id: MsaParsed(
            msa=np.full((paired_rows_index.shape[0], seq.shape[-1]), "-"),
            deletion_matrix=np.zeros(
                (paired_rows_index.shape[0], seq.shape[-1]),
                dtype=int,
            ),
            metadata=pd.DataFrame(),
        )
        for rep_id, seq in msa_collection.rep_seq_map.items()
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


def create_paired(
    msa_collection: MsaCollection, paired_row_cutoff: int
) -> Optional[dict[str, MsaParsed]]:
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
    uniprot_hits = extract_uniprot_hits(msa_collection)

    # Ensure there are at least two chains with UniProt hits after filtering
    if len(uniprot_hits) <= 1:
        # no_uniprot_hits = set(msa_collection.chain_rep_map.values()).difference(
        #     uniprot_hits.keys()
        # )
        # logger.info(
        #     "Skipping MSA pairing: %d chain(s) present. No uniprot hits for %s.",
        #     len(uniprot_hits),
        #     ", ".join(list(no_uniprot_hits)),
        # )
        return None

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
        # logger.info(
        #     "Skipping MSA pairing: No valid pairs for %s.",
        #     ", ".join(list(uniprot_hits.keys())),
        # )
        return None

    # Find species indices that pair rows
    paired_rows_index, missing_rows_index = find_pairing_indices(
        count_array,
        pairing_masks,
        paired_row_cutoff,
    )

    # Map species indices back to MSA row indices
    paired_msa_per_chain = map_to_paired_msa_per_chain(
        msa_collection, uniprot_hits, paired_rows_index, missing_rows_index, species
    )

    return paired_msa_per_chain


def expand_paired_msas(
    msa_collection: MsaCollection, paired_msa_per_chain: dict[str, MsaParsed]
) -> dict[str, MsaParsed]:
    """Creates a dict of MSA objects from paired MSA arrays.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing the paired MSAs and paired
            deletion matrices. Metadata fields are empty.

    Returns:
        dict[str, Msa]:
            A dict of Msa objects containing paired sequences and deletion matrices for
            each unique chain instantiation.
    """
    # # Initialize paired MSA object
    # num_rows = paired_msa_per_chain[next(iter(paired_msa_per_chain))].msa.shape[0]
    # num_cols = sum(
    #     [msa_collection.num_cols[v]
    # for (_, v) in msa_collection.chain_rep_map.items()]
    # )
    # paired_msa = Msa(
    #     msa=np.full((num_rows, num_cols), "-"),
    #     deletion_matrix=np.zeros((num_rows, num_cols), dtype=int),
    #     metadata=pd.DataFrame(),
    # )

    # Update paired MSA with paired data for each chain using the representatives
    # col_offset = 0
    paired_msas = {}
    for chain_id, rep_id in msa_collection.chain_rep_map.items():
        rep_paired_msa = paired_msa_per_chain[rep_id]

        # n_col_paired_i = rep_paired_msa.msa.shape[1]

        # # Replace slices in msa and deletion matrix
        # paired_msa.msa[:, col_offset : (col_offset + n_col_paired_i)] = (
        #     rep_paired_msa.msa
        # )
        # paired_msa.deletion_matrix[:, col_offset : (col_offset + n_col_paired_i)] = (
        #     rep_paired_msa.deletion_matrix
        # )

        # col_offset += n_col_paired_i

        paired_msas[chain_id] = MsaParsed(
            msa=rep_paired_msa.msa,
            deletion_matrix=rep_paired_msa.deletion_matrix,
            metadata=pd.DataFrame(),
        )

    return paired_msas


def create_main(
    msa_collection: MsaCollection,
    paired_msa_per_chain: Union[dict[str, MsaParsed], None],
    aln_order: list[str],
) -> dict[str, MsaParsed]:
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
    for rep_id, chain_data in msa_collection.rep_msa_map.items():
        chain_data = msa_collection.rep_msa_map[rep_id]

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

        rep_main_msas[rep_id] = MsaParsed(
            msa=main_msa_redundant[is_unique, :],
            deletion_matrix=main_deletion_matrix_redundant[is_unique, :],
            metadata=pd.DataFrame(),
        )

    # Reindex dict from representatives to chain IDs
    main_msas = {}
    for chain_id, rep_id in msa_collection.chain_rep_map.items():
        main_msas[chain_id] = rep_main_msas[rep_id]

    return main_msas


def create_crop_to_seq_map(
    atom_array: AtomArray, data_cache_entry_chains: dict[int, Union[int, str]]
) -> MsaSlice:
    """Creates a mapping from the crop to the sequences in the MSA.

    This function connects structure sample processing to MSA sample processing. It
    finds the subset of chains in the crop and creates two mappings, one from the
    chain IDs to representative IDs and another from token IDs to residue IDs for
    tokens in the crop.

    Args:
        atom_array (AtomArray):
            AtomArray of the cropped structure.
        data_cache_entry_chains (dict[int, Union[int, str]]):
            Dictionary of chains to chain features from the data cache.

    Returns:
        MsaSlice:
            Object containing the mappings from the crop to the MSA sequences.
    """
    # Get set of chain IDs in the crop and map to pdb ID + chain ID
    # Subset to protein and RNA chains
    atom_array_with_aln_in_crop = atom_array[
        np.isin(
            atom_array.molecule_type_id,
            [MoleculeType.PROTEIN, MoleculeType.RNA],
        )
    ]
    chain_ids_in_crop = list(set(atom_array_with_aln_in_crop.chain_id))

    chain_rep_map = {}
    tokens_in_chain = {}
    chain_to_molecule_type = {}
    for chain_id_in_crop in chain_ids_in_crop:
        # Get atom array for chain
        atom_array_with_aln_in_crop_chain = atom_array_with_aln_in_crop[
            atom_array_with_aln_in_crop.chain_id == chain_id_in_crop
        ]
        # # Get chain and representative chain ID
        chain_rep_map[chain_id_in_crop] = data_cache_entry_chains[chain_id_in_crop][
            "alignment_representative_id"
        ]

        # Create token -> residue map
        # Note: some atomized residues get duplicate columns from the alignment
        # as they are coming from the same residue
        token_starts = get_token_starts(atom_array_with_aln_in_crop_chain)
        tokens_in_chain[chain_id_in_crop] = {
            token: residue
            for token, residue in zip(
                atom_array_with_aln_in_crop_chain[token_starts].token_id,
                atom_array_with_aln_in_crop_chain[token_starts].res_id - 1,
            )
        }
        chain_to_molecule_type[chain_id_in_crop] = data_cache_entry_chains[
            chain_id_in_crop
        ]["molecule_type"]
    return MsaSlice(
        chain_rep_map=chain_rep_map,
        tokens_in_chain=tokens_in_chain,
        chain_to_molecule_type=chain_to_molecule_type,
    )


def calculate_column_counts(msa_col: np.ndarray, mol_type: MoleculeType) -> np.ndarray:
    """Calculates the counts of residues in an MSA column.

    Args:
        msa_col (np.ndarray):
            Columns slice from an MSA array.
        mol_type (MoleculeType):
            The molecule type of the MSA.

    Returns:
        np.ndarray:
            The counts of residues in the column indexed by the
            STANDARD_RESIDUES_WITH_GAP_1 alphabet.
    """
    # Get correct sub-alphabet, unknown residuem and sort indices for the molecule type
    mol_alphabet = MOLECULE_TYPE_TO_RESIDUES_1[mol_type]
    mol_alphabet_sort_ids = MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1[mol_type]
    mol_alphabet_sorted = mol_alphabet[mol_alphabet_sort_ids]
    res_unknown = MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[mol_type]

    # Get unique residues and counts
    res_unique, res_unique_counts = np.unique(msa_col, return_counts=True)

    # Find which residues are in the molecule alphabet and counts for unknown residues
    res_mol_alphabet_counts = np.zeros(len(mol_alphabet), dtype=int)
    is_in_alphabet = np.isin(res_unique, mol_alphabet)
    res_in_alphabet = res_unique[is_in_alphabet]
    res_unknown_counts = res_unique_counts[~is_in_alphabet]

    # Get indices of residues in and missing from alphabet and "un-sort" them
    id_res_in_alphabet = mol_alphabet_sort_ids[
        np.searchsorted(mol_alphabet_sorted, res_in_alphabet)
    ]
    id_res_unknown = mol_alphabet_sort_ids[
        np.searchsorted(mol_alphabet_sorted, res_unknown)
    ]

    # Assign counts to each character in the un-sorted alphabet
    res_mol_alphabet_counts[id_res_in_alphabet] = res_unique_counts[is_in_alphabet]
    res_mol_alphabet_counts[id_res_unknown] += np.sum(res_unknown_counts)

    # Map molecule alphabet counts to the full residue alphabet
    res_full_alphabet_counts = np.zeros(len(STANDARD_RESIDUES_WITH_GAP_1), dtype=int)
    molecule_alphabet_indices = MOLECULE_TYPE_TO_RESIDUES_POS[mol_type]
    res_full_alphabet_counts[molecule_alphabet_indices] = res_mol_alphabet_counts

    return res_full_alphabet_counts


def apply_crop_to_msa(
    atom_array: AtomArray,
    msa_processed_collection: MsaProcessedCollection,
    msa_slice: MsaSlice,
    token_budget: int,
    max_rows_paired: int,
) -> MsaFeaturePrecursorAF3:
    """Applies crop to the MSA arrays or creates empty MSA arrays.

    Note: this is a temporary connector function between MSA sample processing and
    featurization and will be updated in a future version.

    Args:
        atom_array (AtomArray):
            AtomArray of the cropped structure.
        msa_processed_collection (MsaProcessedCollection):
            Collection of processed MSA data per chain.
        msa_slice (MsaSlice):
            Object containing the mappings from the crop to the MSA sequences.
        token_budget (int):
            The number of tokens in the crop.
        max_rows_paired (int):
            The maximum number of rows to pair.

    Returns:
        MsaProcessed:
            Processed MSA arrays for the crop to featurize.
    """
    # TODO clean up this function
    if msa_processed_collection.query_sequences is not None:
        # Paired MSA rows
        if msa_processed_collection.paired_msas is not None:
            n_rows_paired = msa_processed_collection.paired_msas[
                next(iter(msa_processed_collection.paired_msas))
            ].msa.shape[0]
            n_rows_paired_cropped = min(max_rows_paired, n_rows_paired)
        else:
            n_rows_paired_cropped = 0

        # Main MSA rows
        n_rows_main_per_chain = {
            k: v.msa.shape[0] for k, v in msa_processed_collection.main_msas.items()
        }
        n_rows_main = np.max(list(n_rows_main_per_chain.values()))
        # n_rows_main_cropped = np.min(
        #     [max_rows - (n_rows_paired_cropped + 1), n_rows_main]
        # )
        # n_rows_main_cropped_per_chain = [np.min([n_rows_main_cropped, i])
        # for i in n_rows_main_per_chain.values()]

        # Processed MSA rows
        n_rows = (
            1 + n_rows_paired_cropped + n_rows_main
        )  # instead of n_rows_main_cropped
        msa_processed = np.full([n_rows, token_budget], "-")
        deletion_matrix_processed = np.zeros([n_rows, token_budget])
        msa_mask = np.zeros([n_rows, token_budget])
        msa_profile = np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)])
        deletion_mean = np.zeros(token_budget)

        # Token ID -> token position map
        token_starts = get_token_starts(atom_array)
        token_positions = {
            token: position
            for position, token in enumerate(atom_array[token_starts].token_id)
        }

        # Assign sequence data to corresponding processed MSA slices
        for chain_id, token_res_map in msa_slice.tokens_in_chain.items():
            # Query sequence "MSA"
            q = msa_processed_collection.query_sequences[chain_id]
            # Paired MSA
            if msa_processed_collection.paired_msas is not None:
                p = msa_processed_collection.paired_msas[chain_id]
            # Main MSA
            m = msa_processed_collection.main_msas[chain_id]
            n_rows_main_i = n_rows_main_per_chain[chain_id]

            mol_type = msa_slice.chain_to_molecule_type[chain_id]

            # Iterate over protein/RNA tokens in the crop
            for token_id, res_id in token_res_map.items():
                # Get token column index in the processed MSA
                token_position = token_positions[token_id]
                # Assign row/col slice to the processed MSA slice
                # Query sequence
                msa_processed[0, token_position] = q.msa[res_id]
                deletion_matrix_processed[0, token_position] = q.deletion_matrix[res_id]
                msa_mask[0, token_position] = 1

                # Paired MSA
                if (msa_processed_collection.paired_msas is not None) | (
                    n_rows_paired_cropped != 0
                ):
                    msa_processed[1 : 1 + n_rows_paired_cropped, token_position] = (
                        p.msa[:n_rows_paired_cropped, res_id]
                    )
                    deletion_matrix_processed[
                        1 : 1 + n_rows_paired_cropped, token_position
                    ] = p.deletion_matrix[:n_rows_paired_cropped, res_id]
                    msa_mask[1 : 1 + n_rows_paired_cropped, token_position] = 1

                # Main MSA
                row_offset_main = 1 + n_rows_paired_cropped
                msa_processed[
                    row_offset_main : row_offset_main + n_rows_main_i,
                    token_position,
                ] = m.msa[:, res_id]
                deletion_matrix_processed[
                    row_offset_main : row_offset_main + n_rows_main_i,
                    token_position,
                ] = m.deletion_matrix[:, res_id]
                msa_mask[
                    row_offset_main : row_offset_main + n_rows_main_i,
                    token_position,
                ] = 1

                # If main MSA is empty, leave profile and deletion mean as all-zeros
                if n_rows_main_i != 0:
                    msa_profile[token_position, :] = (
                        calculate_column_counts(
                            m.msa[:, res_id], MoleculeType[mol_type]
                        )
                        / m.msa.shape[0]
                    )
                    deletion_mean[token_position] = np.mean(
                        m.deletion_matrix[:, res_id]
                    )

    else:
        # When there are no protein or RNA chains
        n_rows = 1
        n_rows_paired_cropped = 0
        msa_processed = np.full([n_rows, token_budget], "-")
        deletion_matrix_processed = np.zeros([n_rows, token_budget])
        msa_mask = np.zeros([n_rows, token_budget])
        msa_profile = np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)])
        deletion_mean = np.zeros(token_budget)

    return MsaFeaturePrecursorAF3(
        msa=msa_processed,
        deletion_matrix=deletion_matrix_processed,
        metadata=pd.DataFrame(),
        n_rows_paired=n_rows_paired_cropped + 1,
        msa_mask=msa_mask,
        msa_profile=msa_profile,
        deletion_mean=deletion_mean,
    )
