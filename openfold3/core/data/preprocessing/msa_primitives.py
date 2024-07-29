from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from openfold3.core.data.preprocessing.io import Msa, MsaCollection, parse_msas_sample


def extract_uniprot_hits(msa_collection: MsaCollection) -> dict[str, Msa]:
    """Parses out UniProt Msa objects for unique protein chains from the MsaCollection.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[str, Msa]:
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.
    """
    chain_ids = msa_collection.rep_msa_map.keys()
    # Get uniprot hits
    return {
        chain_id: (
            msa_collection.rep_msa_map[chain_id]["uniprot_hits"]
            if "uniprot_hits" in msa_collection.rep_msa_map[chain_id]
            else msa_collection.rep_msa_map[chain_id]["uniprot"]
        )
        for chain_id in chain_ids
    }


def process_uniprot_metadata(msa: Msa) -> None:
    """Reformats the metadata of an Msa object parsed from a UniProt MSA.

    This function expects a header format of the form:
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


def calculate_distance_to_query(msa: Msa) -> None:
    """Calculates distance to query subsequence for each sequence in the MSA.

    Adds a distance_to_query column to the header DataFrame of an Msa object.

    Args:
        msa (Msa): parsed Msa object
    """
    msa_array = msa.msa
    msa.metadata["distance_to_query"] = np.sum(msa_array == msa_array[0, :], axis=-1)[
        1:
    ] / float(sum(msa_array[0, :] != "-"))


def count_species_per_chain(
    uniprot_hits: dict[str, Msa],
) -> tuple[NDArray[np.int32], list[str]]:
    """Counts the occurrences of sequences from species in each chain's UniProt MSA.

    Args:
        uniprot_hits (dict[str, Msa]):
            Dict mapping chain IDs to Msa objects containing UniProt MSAs.

    Returns:
        NDArray[np.int32], list[str]:
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
    count_array: NDArray[np.int32], mask_keys: Sequence[str]
) -> NDArray[np.bool_]:
    """Generates masks for the pairing process.

    Useful for excluding things like species that occur only in on chain (will not be
    pairable) or species that occur too frequently in the MSA of a single chain (as done
    in the AF2-Multimer pairing code).

    Args:
        count_array (NDArray[np.int32]):
            The array of species occurrence counts per chain.
        mask_keys (Sequence[str]):
            List of strings indicating which mask to add.

    Returns:
        NDArray[np.bool_]: The union of all masks to apply during pairing.
    """
    pairing_masks = np.ones(count_array.shape[1], dtype=bool)

    if "shared_by_two" in mask_keys:
        # Find species that are shared by at least two chains
        pairing_masks = pairing_masks & np.sum(count_array != 0, axis=0) > 1

    if "less_than_600" in mask_keys:
        # Find species that occur more than 600 times in any single chain
        pairing_masks = (
            pairing_masks & np.sum(count_array <= 600, axis=0) == count_array.shape[0]
        )

    return pairing_masks


def find_pairing_indices(
    count_array: NDArray[np.int32],
    pairing_masks: NDArray[np.bool_],
    paired_row_cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The main function for finding indices that pair rows in the MSA arrays.

    This function follows the AF2-Multimer strategy for pairing rows of UniProt MSAs but
    excludes block-diagonal elements (unpaired rows).

    Args:
        count_array (NDArray[np.int32]):
            The array of species occurrence counts per chain
        pairing_masks (NDArray[np.bool_]):
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


def create_paired(
    msa_collection: MsaCollection, paired_row_cutoff: int
) -> NDArray[np.str_]:
    """Creates paired MSA arrays from UniProt MSAs.

    Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on species
    IDs, but excludes the block-diagonal rows, i.e. only includes rows which can be at
    least partially paired.

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
        calculate_distance_to_query(uniprot_hits[chain_id]) for chain_id in uniprot_hits
    ]

    # Count species occurences per chain
    count_array, species = count_species_per_chain(uniprot_hits)

    # Get pairing masks
    pairing_masks = get_pairing_masks(count_array, ["shared_by_two", "less_than_600"])

    # Find indices that pair rows as outlined in the AF3 SI of the AF2-Multimer strategy
    paired_rows_index, missing_rows_index = find_pairing_indices(
        count_array,
        pairing_masks,
        paired_row_cutoff,
    )

    # Sort by similarity to query

    # Map indices to string arrays with gaps

    # Return string array

    return


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


def prepare_msas(
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
        pass

    # Create unpaired non-UniProt MSA arrays

    # Crop MSA arrays

    # Merge MSA arrays
