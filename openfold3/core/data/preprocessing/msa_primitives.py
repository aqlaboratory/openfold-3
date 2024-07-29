from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from openfold3.core.data.preprocessing.io import Msa, MsaCollection, parse_msas_sample


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


def find_pairing_indices(series_list) -> None:
    # Combine into single dataframe
    max_len = max(len(s) for s in series_list)
    df = pd.DataFrame({i: pd.Series(s) for i, s in enumerate(series_list)}).fillna('NaN')

    # Stack the DataFrame to count occurrences
    stacked = df.stack()
    counts = stacked.value_counts()

    # Determine the maximum count of occurrences
    max_occurrence = max(counts)

    # Prepare a dictionary to hold indices for each count
    occurrence_dict = {i: [] for i in range(1, max_occurrence + 1)}

    # Populate the dictionary with elements and their counts
    for elem, count in counts.items():
        if elem == 'NaN':
            continue
        indices = [(s_idx, series_list[s_idx][series_list[s_idx] == elem].index.tolist()) for s_idx in range(len(series_list)) if elem in series_list[s_idx].values]
        occurrence_dict[count].extend(indices)

    sorted_indices = []
    for count in range(max_occurrence, 0, -1):
        elements = occurrence_dict[count]
        sorted_indices.extend(elements)


def find_shared_species(uniprot_hits: dict[str, Msa],
                        paired_row_cutoff: int) -> None:
    """Finds species shared between at least two unique protein chains.

    Adds an is_shared_species column to the header DataFrame of an Msa object,
    indicating if the row corresponds a sequence from a species that occurs in
    at least two UNIQUE chains.

    Args:
        uniprot_hits (dict[str, Msa]): 
            Dictionary mapping representative chain IDs to parsed UniProt Msa objects.
    """

    chain_ids = uniprot_hits.keys()
    n_unique_chains = len(chain_ids)

    # Process uniprot headers and calculate distance to query
    _ = [process_uniprot_metadata(uniprot_hits[chain_id]) for chain_id in chain_ids]
    _ = [calculate_distance_to_query(uniprot_hits[chain_id]) for chain_id in chain_ids]

    species = []
    _ = [species.extend(set(uniprot_hits[chain_id].metadata["species_id"])) for chain_id in chain_ids]
    species = list(set(species))
    species_index = np.arange(len(species))
    species_index_map = {species[i]: i for i in species_index}

    # Get lists of species per chain
    species_index_per_chain = [np.array(uniprot_hits[chain_id].metadata["species_id"].apply(lambda x: species_index_map[x])) for chain_id in chain_ids]

    # Combine all lists into one array
    all_lists = np.concatenate(species_index_per_chain)

    # List to keep track of which list each element came from
    list_indices = np.concatenate([np.array([i]*len(lst)) for i, lst in enumerate(species_index_per_chain)])

    # Get unique integers and their inverse mapping to reconstruct original array
    unique_integers, inverse_indices = np.unique(all_lists, return_inverse=True)

    # Initialize the 2D array to count occurrences
    count_array = np.zeros((n_unique_chains, len(unique_integers)), dtype=int)

    # Use np.add.at for unbuffered in-place addition
    np.add.at(count_array, (list_indices, inverse_indices), 1)

    # Find species that are shared by at least two chains
    is_shared_species = np.sum(count_array != 0, axis=0) > 1

    # Find species that occur more than 600 times in any single chain
    is_nonprevalent_species = np.sum(count_array <= 600, axis=0) == n_unique_chains

    # Apply filters
    count_array_filtered = count_array[:, is_shared_species & is_nonprevalent_species]
    species_index_filtered = species_index[is_shared_species & is_nonprevalent_species]
    is_in_chain_per_species = count_array != 0

    # Iterative row-subtraction
    paired_species_rows = []  # species indices in the MSA feature format
    missing_species_rows = []  # mask for missing species per chain
    n_rows = 0  # number of paired rows
    for n in np.arange(1, n_unique_chains + 1)[::-1]:
        # Find which species are shared by exactly n chains
        is_in_n_chains = sum(count_array_filtered != 0) == n

        # Find lowest number of sequences across chains from shared species
        min_in_n_chains = np.min(count_array_filtered[:, is_in_n_chains], axis=0)

        # Expand filtered species index across occurrences and chains
        cols = np.tile(np.repeat(species_index_filtered, min_in_n_chains), (n_unique_chains, 1))
        n_rows += cols.shape[1]
        paired_species_rows.append(cols.T)

        # Create mask for elements where species is missing form chain
        missing_species_mask = is_in_chain_per_species[:, cols[0, :]]
        missing_species_rows.append(missing_species_mask.T)

        # Subtract min per row
        count_array_filtered -= min_in_n_chains

        # Break if row cutoff reached or n is 1 since we are excluding block-diagonal elements
        if (n_rows >= paired_row_cutoff) | (n == 1):
            break

    # Used paired_species_rows indices to return rows from MSA arrays
    # and mask using missing_species_rows

    # Return an array of paired rows of string arrays
    # Q: does the gap due to missing species need to be different from gap due to alignment?


def create_paired(msa_collection: MsaCollection) -> None:
    """Creates paired MSA arrays from UniProt MSAs.

        Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on
        species IDs, but excludes the block-diagonal rows.

    Args:
        msa_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.
    """
    chain_ids = msa_collection.rep_msa_map.keys()
    # Get uniprot hits
    uniprot_hits = {
        chain_id: (
            msa_collection.rep_msa_map[chain_id]["uniprot_hits"]
            if "uniprot_hits" in msa_collection.rep_msa_map[chain_id]
            else msa_collection.rep_msa_map[chain_id]["uniprot"]
        )
        for chain_id in chain_ids
    }
    # Process uniprot headers and calculate distance to query
    _ = [process_uniprot_metadata(uniprot_hits[chain_id]) for chain_id in chain_ids]
    _ = [calculate_distance_to_query(uniprot_hits[chain_id]) for chain_id in chain_ids]

    # Find shared species
    find_shared_species(uniprot_hits)

    # TODO CONTINUE FROM HERE
    """
    How to go from metadata of distances to query and shared species to
    pairing
    sorting
    concatenating
    """
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
