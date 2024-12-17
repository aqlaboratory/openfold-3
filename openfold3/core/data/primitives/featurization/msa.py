"""This module contains building blocks for MSA feature generation."""

import dataclasses

import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import MsaArray, MsaArrayCollection
from openfold3.core.data.resources.residues import (
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_POS,
    MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1,
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
)


@dataclasses.dataclass(frozen=False)
class MsaFeaturePrecursorAF3:
    """Class representing the fully processed MSA arrays of an assembly.

    Attributes:
        msa (np.array):
            A 2D numpy array containing the aligned sequences.
        deletion_matrix (np.array):
            A 2D numpy array containing the cumulative deletion counts up to each
            position for each row in the MSA.
        n_rows_paired (int):
            Number of paired rows in the MSA array
        msa_mask (np.array):
            A 2D numpy array containing the mask for the MSA.
        msa_profile (np.array):
            A 2D numpy array containing the profile of the MSA.
        deletion_mean (np.array):
            A 1D numpy array containing the mean deletion counts for each row in the
            MSA.
    """

    msa: np.ndarray[str]
    deletion_matrix: np.ndarray[int]
    n_rows_paired: int
    msa_mask: np.ndarray
    msa_profile: np.ndarray
    deletion_mean: np.ndarray


@dataclasses.dataclass(frozen=False)
class MsaTokenMapper:
    """Dataclass storing the token positions and residue IDs of a chain.

    Attributes:
        chain_token_positions (np.ndarray):
            The token positions of the chain.
        res_id (np.ndarray):
            The residue IDs of the chain. Note that these are 1-based.
    """

    chain_token_positions: np.ndarray[int]
    res_id: np.ndarray[int]


@log_runtime_memory(runtime_dict_key="runtime-msa-feat-precursor-rowcount")
def calculate_row_counts(
    msa_array_collection: MsaArrayCollection, max_rows: int, max_rows_paired: int
) -> None:
    """Calculates the row counts of the MSA arrays in the crop/assembly.

    Follows the logic of the AF3 SI sections 2.2 and 2.3.

    Args:
        msa_array_collection (MsaArrayCollection):
            The processed collection of MSA arrays.
        max_rows (int):
            The maximum number of rows allowed for the sum of the cropped paired rows,
            max of main rows + 1 (query).
        max_rows_paired (int):
            The maximum number of rows allowed for the paired MSA.
    """
    if bool(msa_array_collection.chain_id_to_query_seq):
        # Paired MSA rows
        if bool(msa_array_collection.chain_id_to_paired_msa):
            n_rows_paired = next(
                iter(msa_array_collection.chain_id_to_paired_msa.values())
            ).msa.shape[0]
            n_rows_paired_cropped = min(max_rows_paired, n_rows_paired)
        else:
            n_rows_paired_cropped = 0

        # Main MSA rows
        n_rows_main = {
            k: v.msa.shape[0]
            for k, v in msa_array_collection.chain_id_to_main_msa.items()
        }
        n_rows_main_max = max(n_rows_main.values())

        # Combine
        n_rows = min([1 + n_rows_paired_cropped + n_rows_main_max, max_rows])
    else:
        n_rows = 0
        n_rows_paired_cropped = 0
        n_rows_main = {}

    msa_array_collection.set_state_prefeaturized(
        n_rows=n_rows, n_rows_paired=n_rows_paired_cropped, n_rows_main=n_rows_main
    )


def calculate_profile_per_column(
    msa_array: np.ndarray, mol_type: MoleculeType
) -> np.ndarray:
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
    n_col = msa_array.shape[1]

    # Get correct sub-alphabet, unknown residuem and sort indices for the molecule type
    mol_alphabet = MOLECULE_TYPE_TO_RESIDUES_1[mol_type]
    mol_alphabet_sort_ids = MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1[mol_type]
    mol_alphabet_sorted = mol_alphabet[mol_alphabet_sort_ids]
    res_unknown = MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[mol_type]

    # Get unique residues and counts
    res_full_alphabet_counts = np.zeros(
        [n_col, len(STANDARD_RESIDUES_WITH_GAP_1)], dtype=int
    )
    for col_idx in range(n_col):
        col = msa_array[:, col_idx]
        res_unique, res_unique_counts = np.unique(col, return_counts=True, axis=0)

        # Find which residues are in the molecule alphabet and counts for unknown
        # residues
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
        molecule_alphabet_indices = MOLECULE_TYPE_TO_RESIDUES_POS[mol_type]
        res_full_alphabet_counts[col_idx, molecule_alphabet_indices] = (
            res_mol_alphabet_counts
        )

    return res_full_alphabet_counts / msa_array.shape[0]


@log_runtime_memory(
    runtime_dict_key="runtime-msa-feat-precursor-profile-del-mean", multicall=True
)
def calculate_profile_del_mean(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the profile and mean deletion counts for a chain.

    Args:
        msa_array_collection (MsaArrayCollection):
            The processed and pre-featurized collection of MSA arrays.
        chain_id (str):
            The chain ID of the chain to calculate the profile and mean deletion counts
            for.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            The profile and mean deletion counts for the chain.
    """
    # TODO this function is the main runtime bottleneck in the current data pipeline
    # add runtime optimizations
    if bool(msa_array_collection["n_rows_main"][chain_id]):
        profile = calculate_profile_per_column(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa,
            MoleculeType[msa_array_collection.chain_id_to_mol_type[chain_id]],
        )
        del_mean = np.mean(
            msa_array_collection.chain_id_to_main_msa[chain_id].deletion_matrix, axis=0
        )
    else:
        profile = np.zeros(
            [
                msa_array_collection.chain_id_to_main_msa[chain_id].msa.shape[1],
                len(STANDARD_RESIDUES_WITH_GAP_1),
            ],
        )
        del_mean = np.zeros(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa.shape[1]
        )

    return profile, del_mean


@log_runtime_memory(
    runtime_dict_key="runtime-msa-feat-precursor-crop-vstack", multicall=True
)
def crop_vstack_msa_arrays(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
) -> tuple[MsaArray, np.ndarray]:
    """Crops and vertically stacks the MSA arrays for a chain.

    Note: at this point the paired MSA is already cropped for all chains -
    this function only crops the main MSA for a given chain.

    Args:
        msa_array_collection (MsaArrayCollection):
            The processed and pre-featurized collection of MSA arrays.
        chain_id (str):
            The chain ID of the chain to crop and stack the MSA arrays for.

    Returns:
        tuple[MsaArray, np.ndarray]:
            The vertically stacked MSA array and the mask for the MSA.
    """
    # Query stays the same
    msa_array_vstack = msa_array_collection.chain_id_to_query_seq[chain_id]

    # Paired MSA is cropped
    if msa_array_collection["n_rows_paired_cropped"] > 0:
        msa_array_paired_cropped = msa_array_collection.chain_id_to_paired_msa[
            chain_id
        ].truncate(msa_array_collection["n_rows_paired_cropped"])
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_paired_cropped, axis=0
        )

    # This is where the BANDED subsampling logic will go optionally
    # Main MSA is cropped
    if len(msa_array_collection["n_rows_main"]) > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_main_msa[chain_id].truncate(
                msa_array_collection["n_rows"]
                - (msa_array_collection["n_rows_paired_cropped"] + 1)
            ),
            axis=0,
        )

    # Pad bottom of stacked MSA to max(1 + n paired + n main) across all chains
    # and use padding to also create MSA mask for the chain
    msa_array_vstack, msa_array_vstack_mask = msa_array_vstack.pad(
        target_length=msa_array_collection["n_rows"], axis=0
    )

    return msa_array_vstack, msa_array_vstack_mask


@log_runtime_memory(
    runtime_dict_key="runtime-msa-feat-precursor-create-token-mapper", multicall=True
)
def create_msa_token_mapper(atom_array: AtomArray, chain_id: str) -> MsaTokenMapper:
    """Creates an MsaTokenMapper for a given chain in the assembly.

    Args:
        atom_array (AtomArray):
            Assembly atom array.
        chain_id (str):
            Chain ID of the chain to create the token mapper

    Returns:
        MsaTokenMapper:
            Token mapper for the chain.
    """
    # Token positions
    atom_array_chain = atom_array[atom_array.chain_id == chain_id]
    chain_token_starts = get_token_starts(atom_array_chain)
    chain_token_positions = atom_array_chain[chain_token_starts].token_position

    # Residue IDs and repeats
    res_ids = atom_array_chain[chain_token_starts].res_id

    return MsaTokenMapper(chain_token_positions, res_ids)


@log_runtime_memory(runtime_dict_key="runtime-msa-feat-precursor-map", multicall=True)
def map_msas_to_tokens(
    msa_feature_precursor: MsaFeaturePrecursorAF3,
    msa_array_vstack: MsaArray,
    msa_array_vstack_mask: np.ndarray[int],
    profile: np.ndarray[float],
    del_mean: np.ndarray[float],
    msa_token_mapper: MsaTokenMapper,
) -> None:
    """Maps the processed and stacked MSA array of chain to tokens.

    Args:
        msa_feature_precursor (MsaFeaturePrecursorAF3):
            The pre-allocted MSA feature precursor container.
        msa_array_vstack (MsaArray):
            The vertically stacked MSA array for the chain.
        msa_array_vstack_mask (np.ndarray[int]):
            Mask for padding MSA features of the current chain to the max number
            of rows across all chains.
        profile (np.ndarray[float]):
            The profile of the MSA - this is calculated based on the uncropped main MSA
            only.
        del_mean (np.ndarray[float]):
            The mean of deletion counts for each column in the MSA of the current chain.
            This is calculated based on the uncropped main MSA only.
        msa_token_mapper (MsaTokenMapper):
            Token mapper for the chain.
    """
    # Unpack token mapper
    token_positions = msa_token_mapper.chain_token_positions
    msa_column_positions = msa_token_mapper.res_id - 1

    # Map MSA data to tokens
    # Expands column positions for atomized tokens
    msa_feature_precursor.msa[:, token_positions] = msa_array_vstack.msa[
        :, msa_column_positions
    ]
    msa_feature_precursor.deletion_matrix[:, token_positions] = (
        msa_array_vstack.deletion_matrix[:, msa_column_positions]
    )
    msa_feature_precursor.msa_mask[:, token_positions] = msa_array_vstack_mask[
        :, msa_column_positions
    ]
    msa_feature_precursor.msa_profile[token_positions] = profile[
        msa_column_positions, :
    ]
    msa_feature_precursor.deletion_mean[token_positions] = del_mean[
        msa_column_positions
    ]


@log_runtime_memory(runtime_dict_key="runtime-msa-feat-precursor")
def create_msa_feature_precursor_af3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
    max_rows: int,
    max_rows_paired: int,
    token_budget: int,
) -> MsaFeaturePrecursorAF3:
    """Creates a set of precursor arrays for AF3 MSA featurization.

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
    if bool(msa_array_collection.chain_id_to_query_seq):
        # fetch rowcounts
        calculate_row_counts(msa_array_collection, max_rows, max_rows_paired)

        # Pre-allocate feature precursor container
        msa_feature_precursor = MsaFeaturePrecursorAF3(
            msa=np.full([msa_array_collection["n_rows"], token_budget], "-"),
            deletion_matrix=np.zeros([msa_array_collection["n_rows"], token_budget]),
            n_rows_paired=msa_array_collection["n_rows_paired_cropped"] + 1,
            msa_mask=np.zeros([msa_array_collection["n_rows"], token_budget]),
            msa_profile=np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(token_budget),
        )

        # Process each chain
        for chain_id in msa_array_collection.chain_id_to_rep_id:
            # Calculate profile and del mean for chain across all main columns
            profile, del_mean = calculate_profile_del_mean(
                msa_array_collection, chain_id
            )

            # Crop and vertically stack query, paired MSA and main MSA arrays
            msa_array_vstack, msa_array_vstack_mask = crop_vstack_msa_arrays(
                msa_array_collection,
                chain_id,
            )
            # Get token positions and repeats from the atomarray of the chain
            msa_token_mapper = create_msa_token_mapper(atom_array, chain_id)

            # Map to tokens
            map_msas_to_tokens(
                msa_feature_precursor,
                msa_array_vstack,
                msa_array_vstack_mask,
                profile,
                del_mean,
                msa_token_mapper,
            )

    else:
        # When there are no protein or RNA chains
        msa_feature_precursor = MsaFeaturePrecursorAF3(
            msa=np.full([1, token_budget], "-"),
            deletion_matrix=np.zeros([1, token_budget]),
            n_rows_paired=1,
            msa_mask=np.zeros([1, token_budget]),
            msa_profile=np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(token_budget),
        )

    return msa_feature_precursor
