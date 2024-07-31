from typing import Sequence

import numpy as np
import pandas as pd

from openfold3.core.data.preprocessing.io import Msa, MsaCollection, parse_msas_sample


def extract_uniprot_hits(msa_collection: MsaCollection) -> dict[str, Msa]:
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
    chain_ids = msa_collection.rep_msa_map.keys()
    # Get uniprot hits, exclude MSAs only with query
    return {
        chain_id: msa_collection.rep_msa_map[chain_id]["uniprot_hits"]
        for chain_id in chain_ids
        if len(msa_collection.rep_msa_map[chain_id]["uniprot_hits"]) > 1
    }


def process_uniprot_metadata(msa: Msa) -> None:
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
    query sequence, an empty DataFrame is assigned to the metatadata attribute.

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


def sort_msa_by_distance_to_query(msa: Msa) -> None:
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
    uniprot_hits: dict[str, Msa],
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

    Useful for excluding things like species that occur only in on chain (will not be
    pairable) or species that occur too frequently in the MSA of a single chain (as done
    in the AF2-Multimer pairing code).

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

        # Find lowest number of sequences across chains from shared species
        min_in_n_chains = np.min(count_array_filtered[:, is_in_n_chains], axis=0)

        # Expand filtered species index across occurrences and chains
        cols = np.tile(
            np.repeat(species_index_filtered, min_in_n_chains), (n_unique_chains, 1)
        )
        n_rows += cols.shape[1]
        paired_species_rows.append(cols.T)

        # Create mask for elements where species is missing from chain
        missing_species_mask = is_in_chain_per_species[:, cols[0, :]]
        missing_species_rows.append(missing_species_mask.T)

        # Subtract min per row
        count_array_filtered -= min_in_n_chains

        # If row cutoff reached, crop final arrays to the row cutoff and break
        if n_rows >= paired_row_cutoff:
            n_rows_final = paired_row_cutoff - sum(
                [rows.shape[0] for rows in paired_species_rows[:-1]]
            )
            paired_species_rows[-1] = paired_species_rows[-1][: n_rows_final + 1, :]
            missing_species_mask[-1] = missing_species_mask[-1][: n_rows_final + 1, :]
            break

        # Concatenate all paired arrays into a single paired array
        paired_rows_index = np.concatenate(paired_species_rows, axis=0)
        missing_rows_index = np.concatenate(missing_species_rows, axis=0)

    return paired_rows_index, missing_rows_index


def map_to_paired_msa_per_chain(
    uniprot_hits: dict[str, Msa],
    paired_rows_index: np.ndarray[np.int32],
    missing_rows_index: np.ndarray[np.int32],
    species: list,
) -> dict[str, Msa]:
    """Maps paired species indices to MSA row indices.

    Args:
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
    # Pre-allocate MSA objects
    paired_msa_per_chain = {
        chain_id: Msa(
            msa=np.full(
                (paired_rows_index.shape[0], uniprot_hits[chain_id].msa.shape[-1]), "-"
            ),
            deletion_matrix=np.zeros(
                (paired_rows_index.shape[0], uniprot_hits[chain_id].msa.shape[-1]),
                dtype=int,
            ),
            metadata=pd.DataFrame(),
        )
        for chain_id in uniprot_hits
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


def merge_paired_msas(
    msa_collection: MsaCollection, paired_msa_per_chain: dict[str, Msa]
) -> Msa:
    """Creates a single MSA object from paired MSA arrays.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing the paired MSAs and paired
            deletion matrices. Metadata fields are empty.

    Returns:
        Msa:
            A single Msa object containing the paired MSA arrays and deletion matrices
            for all processed chain IDs.
    """
    # Initialize paired MSA object
    num_rows = paired_msa_per_chain[next(iter(paired_msa_per_chain))].msa.shape[0]
    num_cols = msa_collection.num_cols
    paired_msa = Msa(
        msa=np.full((num_rows, num_cols), "-"),
        deletion_matrix=np.zeros((num_rows, num_cols), dtype=int),
        metadata=pd.DataFrame(),
    )

    # Update paired MSA with paired data for each chain using the representatives
    col_offset = 0
    for chain_id in msa_collection.chain_rep_map:
        rep_id = msa_collection.chain_rep_map[chain_id]
        rep_paired_msa = paired_msa_per_chain[rep_id]
        n_col_paired_i = rep_paired_msa.msa.shape[1]

        # Replace slices in msa and deletion matrix
        paired_msa.msa[:, col_offset : (col_offset + n_col_paired_i)] = (
            rep_paired_msa.msa
        )
        paired_msa.deletion_matrix[:, col_offset : (col_offset + n_col_paired_i)] = (
            rep_paired_msa.deletion_matrix
        )

        col_offset += n_col_paired_i

    return paired_msa


def create_paired(
    msa_collection: MsaCollection, paired_row_cutoff: int
) -> Msa:
    """Creates paired MSA arrays from UniProt MSAs.

    Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on species
    IDs, but excludes the block-diagonal rows, i.e. only includes rows which can be at
    least partially paired, as suggested by the AF3 SI.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
        paired_row_cutoff (int):
            The maximum number of rows to pair.
    """
    # Get parsed uniprot hits
    uniprot_hits = extract_uniprot_hits(msa_collection)

    # Process uniprot headers and calculate distance to query
    _ = [process_uniprot_metadata(uniprot_hits[chain_id]) for chain_id in uniprot_hits]
    _ = [
        sort_msa_by_distance_to_query(uniprot_hits[chain_id])
        for chain_id in uniprot_hits
    ]

    # Count species occurences per chain
    count_array, species = count_species_per_chain(uniprot_hits)

    # Get pairing masks
    pairing_masks = get_pairing_masks(count_array, ["shared_by_two", "less_than_600"])

    # Find species indices that pair rows
    paired_rows_index, missing_rows_index = find_pairing_indices(
        count_array,
        pairing_masks,
        paired_row_cutoff,
    )

    # Map species indices back to MSA row indices
    paired_msa_per_chain = map_to_paired_msa_per_chain(
        uniprot_hits, paired_rows_index, missing_rows_index, species
    )

    # Expand across duplicate chains and concatenate
    paired_msa = merge_paired_msas(msa_collection, paired_msa_per_chain)

    return paired_msa


def find_monomer_homomer(msa_collection: MsaCollection) -> bool:
    """Determines if the sample is a monomer or homomer.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        bool: Whether the sample is a monomer or a full homomer.
    """
    # Extract chain IDs and representative chain IDs
    chain_rep_map = msa_collection.chain_rep_map
    chain_ids, representative_chain_ids = (
        list(chain_rep_map.keys()),
        list(set(chain_rep_map.values())),
    )
    return len(chain_ids) == 1 | (
        len(representative_chain_ids) == 1 & len(chain_ids) > 1
    )


def create_unpaired(msa_collection: MsaCollection):
    pass


def process_msas(
    chain_ids, alignments_path, use_alignment_database, alignment_index, max_seq_counts
):
    """Prepares the arrays needed to create MSA feature tensors.

    Args:
        chain_ids (_type_): _description_
        alignments_path (_type_): _description_
        use_alignment_database (_type_): _description_
        alignment_index (_type_): _description_
        max_seq_counts (_type_): _description_
    """

    # Parse MSAs for the cropped sample
    msa_collection = parse_msas_sample(
        chain_ids=chain_ids,
        alignments_path=alignments_path,
        use_alignment_database=use_alignment_database,
        alignment_index=alignment_index,
        max_seq_counts=max_seq_counts,
    )

    # Determine whether to do pairing
    is_monomer_homomer = find_monomer_homomer(msa_collection)

    if not is_monomer_homomer:
        # Create paired UniProt MSA arrays
        paired_msa = create_paired(msa_collection, paired_row_cutoff=8191)

    # Create unpaired non-UniProt MSA arrays

    # Crop MSA arrays

    # Merge MSA arrays
