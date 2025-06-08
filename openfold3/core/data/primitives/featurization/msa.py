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
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
    map_str_array_to_idx_array,
)


@dataclasses.dataclass(frozen=False)
class MsaFeaturePrecursorAF3:
    """Class representing the fully processed MSA arrays of an assembly.

    Attributes:
        msa (np.ndarray[str]):
            A 2D numpy array containing the aligned sequences.
        msa_index (np.ndarray[int]):
            A 2D numpy array containing the position of the residues in the global
            molecule alphabet of all molecule types, STANDARD_RESIDUES_WITH_GAP_1.
        deletion_matrix (np.ndarray[int]):
            A 2D numpy array containing the cumulative deletion counts up to each
            position for each row in the MSA.
        n_rows_paired (int):
            Number of paired rows in the MSA array
        msa_mask (np.ndarray[int]):
            A 2D numpy array containing the mask for the MSA.
        msa_profile (np.ndarray[float]):
            A 2D numpy array containing the profile of the MSA.
        deletion_mean (np.ndarray[float]):
            A 1D numpy array containing the mean deletion counts for each row in the
            MSA.
    """

    msa: np.ndarray[str]
    msa_index: np.ndarray[int]
    deletion_matrix: np.ndarray[int]
    n_rows_paired: int
    msa_mask: np.ndarray[int]
    msa_profile: np.ndarray[float]
    deletion_mean: np.ndarray[float]


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
        n_rows=n_rows,
        n_rows_paired_cropped=n_rows_paired_cropped,
        n_rows_main=n_rows_main,
    )


def calculate_profile(
    msa_array: np.ndarray, molecule_type: MoleculeType, chunk_size: int
) -> np.ndarray:
    """Calculates the fractions of residue occurences per character per column.

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

    msa_index = map_str_array_to_idx_array(msa_array, molecule_type)

    n_rows, n_cols = msa_index.shape
    n_symbols = len(STANDARD_RESIDUES_WITH_GAP_1)
    counts = np.zeros((n_cols, n_symbols), dtype=int)
    col_start = 0

    while col_start < n_cols:
        col_end = min(col_start + chunk_size, n_cols)
        msa_chunk = msa_index[:, col_start:col_end]
        block_n_cols = col_end - col_start
        # Flatten subarray (size = n_rows * block_n_cols)
        val_indices = msa_chunk.ravel()  # row-major flatten by default
        # Build local col_indices of the same flattened shape
        col_indices_local = np.repeat(np.arange(block_n_cols), n_rows)
        # Now each col in this chunk is offset from the "absolute" col_start, but for
        # bincount we just care about "relative" indexing from 0...(block_n_cols-1). We
        # combine into a single 1D array: offset + val Where offset = col_indices_local
        # * n_symbols That ensures each column in the chunk has a distinct range in the
        # output
        to_count_local = col_indices_local * n_symbols + val_indices

        # Bincount for this chunk
        # We'll have block_n_cols*n_symbols possible bins
        chunk_counts_1d = np.bincount(
            to_count_local, minlength=block_n_cols * n_symbols
        )

        # Reshape into (block_n_cols, n_symbols)
        chunk_counts_2d = chunk_counts_1d.reshape(block_n_cols, n_symbols)

        # Accumulate into the global array
        counts[col_start:col_end, :] += chunk_counts_2d

        col_start = col_end

    return counts / n_rows


@log_runtime_memory(
    runtime_dict_key="runtime-msa-feat-precursor-profile-del-mean", multicall=True
)
def calculate_profile_del_mean(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
    msa_profile_chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the profile and mean deletion counts for a chain.

    Args:
        msa_array_collection (MsaArrayCollection):
            The processed and pre-featurized collection of MSA arrays.
        chain_id (str):
            The chain ID of the chain to calculate the profile and mean deletion counts
            for.
        msa_profile_chunk_size (int):
            The number of columns to simultaneously calculate the MSA profile for.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            The profile and mean deletion counts for the chain.
    """
    if bool(msa_array_collection.row_counts["n_rows_main"][chain_id]):
        profile = calculate_profile(
            msa_array_collection.chain_id_to_main_msa[chain_id].msa,
            msa_array_collection.chain_id_to_mol_type[chain_id],
            chunk_size=msa_profile_chunk_size,
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
    if msa_array_collection.row_counts["n_rows_paired_cropped"] > 0:
        msa_array_paired_cropped = msa_array_collection.chain_id_to_paired_msa[
            chain_id
        ].truncate(msa_array_collection.row_counts["n_rows_paired_cropped"])
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_paired_cropped, axis=0
        )

    # This is where the BANDED subsampling logic will go optionally
    # Main MSA is cropped
    if len(msa_array_collection.row_counts["n_rows_main"]) > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_main_msa[chain_id].truncate(
                msa_array_collection.row_counts["n_rows"]
                - (msa_array_collection.row_counts["n_rows_paired_cropped"] + 1)
            ),
            axis=0,
        )

    # Pad bottom of stacked MSA to max(1 + n paired + n main) across all chains
    # and use padding to also create MSA mask for the chain
    msa_array_vstack, msa_array_vstack_mask = msa_array_vstack.pad(
        target_length=msa_array_collection.row_counts["n_rows"], axis=0
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
    molecule_type: MoleculeType,
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
        molecule_type (MoleculeType):
            The molecule type of the current chain.
    """
    # Unpack token mapper
    token_positions = msa_token_mapper.chain_token_positions
    msa_column_positions = msa_token_mapper.res_id - 1

    # Map MSA data to tokens
    # Expands column positions for atomized tokens
    msa_array = msa_array_vstack.msa[:, msa_column_positions]
    msa_feature_precursor.msa[:, token_positions] = msa_array
    msa_feature_precursor.msa_index[:, token_positions] = map_str_array_to_idx_array(
        msa_array=msa_array, molecule_type=molecule_type
    )
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
    n_tokens: int | None = None,
) -> MsaFeaturePrecursorAF3:
    """Creates a set of precursor arrays for AF3 MSA featurization.

    Args:
        atom_array (AtomArray):
            AtomArray of the cropped structure.
        msa_processed_collection (MsaProcessedCollection):
            Collection of processed MSA data per chain.
        msa_slice (MsaSlice):
            Object containing the mappings from the crop to the MSA sequences.
        n_tokens (int | None):
            The number of tokens in the crop during training or in the whole structure
            to predict during inference. If None, it will be set to the number of tokens
            in the atom array.
        max_rows_paired (int):
            The maximum number of rows to pair.

    Returns:
        MsaProcessed:
            Processed MSA arrays for the crop during training or in the whole structure
            to predict during inference to featurize.
    """
    # Set n_tokens to the number of tokens in the atom array if not provided
    if n_tokens is None:
        n_tokens = len(get_token_starts(atom_array))

    if bool(msa_array_collection.chain_id_to_query_seq):
        # fetch rowcounts
        calculate_row_counts(msa_array_collection, max_rows, max_rows_paired)

        # Pre-allocate feature precursor container
        msa_feature_precursor = MsaFeaturePrecursorAF3(
            msa=np.full([msa_array_collection.row_counts["n_rows"], n_tokens], "-"),
            msa_index=np.ones([msa_array_collection.row_counts["n_rows"], n_tokens])
            * np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
            deletion_matrix=np.zeros(
                [msa_array_collection.row_counts["n_rows"], n_tokens]
            ),
            n_rows_paired=msa_array_collection.row_counts["n_rows_paired_cropped"] + 1,
            msa_mask=np.zeros([msa_array_collection.row_counts["n_rows"], n_tokens]),
            msa_profile=np.zeros([n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(n_tokens),
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
                msa_feature_precursor=msa_feature_precursor,
                msa_array_vstack=msa_array_vstack,
                msa_array_vstack_mask=msa_array_vstack_mask,
                profile=profile,
                del_mean=del_mean,
                msa_token_mapper=msa_token_mapper,
                molecule_type=msa_array_collection.chain_id_to_mol_type[chain_id],
            )

    else:
        # When there are no protein or RNA chains
        msa_feature_precursor = MsaFeaturePrecursorAF3(
            msa=np.full([1, n_tokens], "-"),
            msa_index=np.ones([1, n_tokens])
            * np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
            deletion_matrix=np.zeros([1, n_tokens]),
            n_rows_paired=1,
            msa_mask=np.zeros([1, n_tokens]),
            msa_profile=np.zeros([n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(n_tokens),
        )

    return msa_feature_precursor
