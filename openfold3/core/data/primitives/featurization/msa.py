"""This module contains building blocks for MSA feature generation."""

import dataclasses

import numpy as np
import pandas as pd
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import MsaArrayCollection
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


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-apply-crop")
def create_msa_feature_precursor_af3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
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
    # TODO clean up this function
    if msa_array_collection.query_sequences is not None:
        # Paired MSA rows
        if msa_array_collection.paired_msas is not None:
            n_rows_paired = msa_array_collection.paired_msas[
                next(iter(msa_array_collection.paired_msas))
            ].msa.shape[0]
            n_rows_paired_cropped = min(max_rows_paired, n_rows_paired)
        else:
            n_rows_paired_cropped = 0

        # Main MSA rows
        n_rows_main_per_chain = {
            k: v.msa.shape[0] for k, v in msa_array_collection.main_msas.items()
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
        # TODO rework with token_position annotation
        token_starts = get_token_starts(atom_array)
        token_positions = {
            token: position
            for position, token in enumerate(atom_array[token_starts].token_id)
        }

        # Assign sequence data to corresponding processed MSA slices
        # !!! msa_array_collection.tokens_in_chain.items() doesn't
        # actually, work, need a way to get token position map
        for chain_id, token_res_map in msa_array_collection.tokens_in_chain.items():
            # Query sequence "MSA"
            q = msa_array_collection.query_sequences[chain_id]
            # Paired MSA
            if msa_array_collection.paired_msas is not None:
                p = msa_array_collection.paired_msas[chain_id]
            # Main MSA
            m = msa_array_collection.main_msas[chain_id]
            n_rows_main_i = n_rows_main_per_chain[chain_id]

            mol_type = msa_array_collection.chain_to_mol_type[chain_id]

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
                if (msa_array_collection.paired_msas is not None) | (
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
