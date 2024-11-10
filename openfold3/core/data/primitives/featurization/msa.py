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


def calculate_row_counts(
    msa_array_collection: MsaArrayCollection, max_rows: int, max_rows_paired: int
) -> tuple[int, int, bool, bool]:
    if len(msa_array_collection.chain_id_to_query_seq) != 0:
        # Paired MSA rows
        if len(msa_array_collection.chain_id_to_paired_msa) != 0:
            n_rows_paired = next(
                iter(msa_array_collection.chain_id_to_paired_msa.values())
            ).msa.shape[0]
            n_rows_paired_cropped = min(max_rows_paired, n_rows_paired)
        else:
            n_rows_paired = 0
            n_rows_paired_cropped = 0

        # Main MSA rows
        n_rows_main = max(
            [v.msa.shape[0] for v in msa_array_collection.chain_id_to_main_msa.values()]
        )

        # Combine
        n_rows = min([1 + n_rows_paired_cropped + n_rows_main, max_rows])

    paired_exists = len(msa_array_collection.chain_id_to_paired_msa) != 0
    main_exists = len(msa_array_collection.chain_id_to_main_msa) != 0

    return n_rows, n_rows_paired_cropped, paired_exists, main_exists


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


def calculate_profile(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
    main_exists: bool,
    profile_all: np.ndarray,
    del_mean_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if main_exists:
        profile = calculate_profile_per_column(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa,
            MoleculeType[msa_array_collection.chain_id_to_mol_type[chain_id]],
        )
        del_mean = np.mean(
            msa_array_collection.chain_id_to_main_msa[chain_id].deletion_matrix, axis=0
        )
    else:
        profile = np.zeros(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa.shape[1],
            len(STANDARD_RESIDUES_WITH_GAP_1),
        )
        del_mean = np.zeros(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa.shape[1]
        )

    return np.concatenate([profile_all, profile]), np.concatenate(
        [del_mean_all, del_mean]
    )


def crop_vstack_msa_arrays(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
    n_rows: int,
    n_rows_paired_cropped: int,
    paired_exists: bool,
    main_exists: bool,
) -> MsaArray:
    msa_array_vstack = msa_array_collection.chain_id_to_query_seq[chain_id]
    if paired_exists:
        msa_array_paired_cropped = msa_array_collection.chain_id_to_paired_msa[
            chain_id
        ].truncate(n_rows_paired_cropped)
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_paired_cropped, axis=0
        )

    # This is where the BANDED subsampling logic will go optionally
    if main_exists:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_main_msa[chain_id].truncate(
                n_rows - (n_rows_paired_cropped + 1)
            ),
            axis=0,
        )

    return msa_array_vstack


def map_msas_to_tokens(
    msa_array_all: MsaArray,
    msa_masks_all: np.ndarray[int],
    profile_all: np.ndarray[float],
    del_mean_all: np.ndarray[float],
    token_positions_all: np.ndarray[int],
    n_rows: int,
    n_rows_paired_cropped: int,
    token_budget: int,
) -> MsaFeaturePrecursorAF3:
    # Pre-allocate containers
    msa_tokenized = np.full([n_rows, token_budget], "-")
    deletion_matrix_tokenized = np.zeros([n_rows, token_budget])
    msa_mask_tokenized = np.zeros([n_rows, token_budget])
    profile_tokenized = np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)])
    del_mean_tokenized = np.zeros(token_budget)

    # Map MSA arrays to tokens
    msa_tokenized[:, token_positions_all] = msa_array_all.msa
    deletion_matrix_tokenized[:, token_positions_all] = msa_array_all.deletion_matrix
    msa_mask_tokenized[:, token_positions_all] = msa_masks_all
    profile_tokenized[token_positions_all] = profile_all
    del_mean_tokenized[token_positions_all] = del_mean_all

    return MsaFeaturePrecursorAF3(
        msa=msa_tokenized,
        deletion_matrix=deletion_matrix_tokenized,
        n_rows_paired=n_rows_paired_cropped,
        msa_mask=msa_mask_tokenized,
        msa_profile=profile_tokenized,
        deletion_mean=del_mean_tokenized,
    )


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-apply-crop")
def create_msa_feature_precursor_af3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
    max_rows: int,
    max_rows_paired: int,
    token_budget: int,
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

    # fetch rowcounts
    if len(msa_array_collection.chain_id_to_query_seq) != 0:
        n_rows, n_rows_paired_cropped, paired_exists, main_exists = (
            calculate_row_counts(msa_array_collection, max_rows, max_rows_paired)
        )

        # Pre-allocate containers
        msa_array_all = MsaArray(
            msa=np.empty((n_rows, 0), dtype=str),
            deletion_matrix=np.empty((n_rows, 0), dtype=str),
        )
        msa_masks_all = np.empty((n_rows, 0), dtype=int)
        token_positions_all = np.empty((0,), dtype=int)
        profile_all = np.empty((0, len(STANDARD_RESIDUES_WITH_GAP_1)), dtype=int)
        del_mean_all = np.empty((0,), dtype=int)

        # Process each chain
        for chain_id in msa_array_collection.chain_id_to_rep_id:
            # Calculate profile and del mean for chain across all main columns
            profile_all, del_mean_all = calculate_profile(
                msa_array_collection, chain_id, main_exists, profile_all, del_mean_all
            )

            # Crop and vertically stack query, paired MSA and main MSA arrays
            msa_array_vstack = crop_vstack_msa_arrays(
                msa_array_collection,
                chain_id,
                n_rows,
                n_rows_paired_cropped,
                paired_exists,
                main_exists,
            )

            # Pad bottom of stacked MSA to max(1 + n paired + n main) across all chains
            # and use padding to also create MSA mask for the chain
            msa_array_vstack, msa_array_vstack_mask = msa_array_vstack.pad(
                target_length=n_rows, axis=0
            )

            # Get token positions from the atomarray of the chain
            atom_array_chain = atom_array[atom_array.chain_id == chain_id]
            chain_token_starts = get_token_starts(atom_array_chain)
            chain_token_positions = atom_array_chain[chain_token_starts].token_position

            # Horizontally stack along all chains
            msa_array_all = msa_array_all.concatenate(msa_array_vstack, axis=1)
            msa_masks_all = np.concatenate(
                [msa_masks_all, msa_array_vstack_mask], axis=1
            )
            token_positions_all = np.concatenate(
                [token_positions_all, chain_token_positions]
            )

            del [
                msa_array_vstack,
                msa_array_vstack_mask,
                chain_token_positions,
                atom_array_chain,
            ]

        # Map all arrays from horizontal stack to tokens
        msa_feature_precursor = map_msas_to_tokens(
            msa_array_all,
            msa_masks_all,
            profile_all,
            del_mean_all,
            token_positions_all,
            n_rows,
            n_rows_paired_cropped,
            token_budget,
        )

    else:
        # When there are no protein or RNA chains
        msa_feature_precursor = MsaFeaturePrecursorAF3(
            msa=np.full([n_rows, token_budget], "-"),
            deletion_matrix=np.zeros([n_rows, token_budget]),
            n_rows_paired=0,
            msa_mask=np.zeros([n_rows, token_budget]),
            msa_profile=np.zeros([token_budget, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(token_budget),
        )

    return msa_feature_precursor
