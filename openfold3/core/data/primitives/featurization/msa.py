# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains building blocks for MSA feature generation."""

import dataclasses

import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
    get_token_starts,
)
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
class MsaFeaturePrecursorOF3:
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


@log_runtime_memory(
    runtime_dict_key="runtime-msa-feat-precursor-vstack", multicall=True
)
def vstack_pad_msa_arrays(
    msa_array_collection: MsaArrayCollection,
    chain_id: str,
) -> tuple[MsaArray, np.ndarray]:
    """Vertically stacks the MSA arrays for a chain and pads to max MSA depth across
    all chains.

    Note: at this point the paired MSA is already cropped for all chains and the
    main MSA is subsampled.

    Args:
        msa_array_collection (MsaArrayCollection):
            The processed and pre-featurized collection of MSA arrays.
        chain_id (str):
            The chain ID of the chain to crop and stack the MSA arrays for.

    Returns:
        tuple[MsaArray, np.ndarray]:
            The vertically stacked MSA array and the mask for the MSA.
    """
    # Query
    msa_array_vstack = msa_array_collection.chain_id_to_query_seq[chain_id]

    # Paired MSA
    if msa_array_collection.row_counts.n_rows_paired_subsampled > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_paired_msa[chain_id], axis=0
        )

    # Main MSA
    if len(msa_array_collection.row_counts.n_rows_main_subsampled) > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_main_msa[chain_id],
            axis=0,
        )

    # Pad bottom of stacked MSA to max(1 + n paired + n main) across all chains
    # and use padding to also create MSA mask for the chain
    msa_array_vstack, msa_array_vstack_mask = msa_array_vstack.pad(
        target_length=msa_array_collection.row_counts.n_rows_total, axis=0
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
    msa_feature_precursor: MsaFeaturePrecursorOF3,
    msa_array_vstack: MsaArray,
    msa_array_vstack_mask: np.ndarray[int] | None,
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
        msa_array_vstack_mask (np.ndarray[int] | None):
            Mask for padding MSA features of the current chain to the max number
            of rows across all chains
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
    if msa_array_vstack_mask is not None:
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
def create_msa_feature_precursor_of3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
    n_tokens: int,
) -> MsaFeaturePrecursorOF3:
    """Creates a set of precursor arrays for AF3 MSA featurization.

    Args:
        atom_array (AtomArray):
            AtomArray of the cropped structure.
        msa_array_collection (MsaArrayCollection):
            Collection of processed MSA data per chain.
        n_tokens (int):
            Number of tokens in the atom array.

    Returns:
        MsaProcessed:
            Processed MSA arrays for the crop during training or in the whole structure
            to predict during inference to featurize.
    """
    if bool(msa_array_collection.chain_id_to_query_seq):
        # Pre-allocate feature precursor container
        msa_feature_precursor = MsaFeaturePrecursorOF3(
            msa=np.full([msa_array_collection.row_counts.n_rows_total, n_tokens], "-"),
            msa_index=np.ones([msa_array_collection.row_counts.n_rows_total, n_tokens])
            * np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
            deletion_matrix=np.zeros(
                [msa_array_collection.row_counts.n_rows_total, n_tokens]
            ),
            n_rows_paired=msa_array_collection.row_counts.n_rows_paired_subsampled + 1,
            msa_mask=np.ones([msa_array_collection.row_counts.n_rows_total, n_tokens]),
            msa_profile=np.zeros([n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(n_tokens),
        )

        # Process each chain
        for chain_id in msa_array_collection.chain_id_to_rep_id:
            # Crop and vertically stack query, paired MSA and main MSA arrays
            msa_array_vstack, _ = vstack_pad_msa_arrays(
                msa_array_collection,
                chain_id,
            )
            # Get token positions and repeats from the atomarray of the chain
            msa_token_mapper = create_msa_token_mapper(atom_array, chain_id)

            # Map to tokens
            map_msas_to_tokens(
                msa_feature_precursor=msa_feature_precursor,
                msa_array_vstack=msa_array_vstack,
                msa_array_vstack_mask=None,
                profile=msa_array_collection.chain_id_to_profile[chain_id],
                del_mean=msa_array_collection.chain_id_to_deletion_mean[chain_id],
                msa_token_mapper=msa_token_mapper,
                molecule_type=msa_array_collection.chain_id_to_mol_type[chain_id],
            )

    else:
        # When there are no protein or RNA chains
        msa_feature_precursor = MsaFeaturePrecursorOF3(
            msa=np.full([1, n_tokens], "-"),
            msa_index=np.ones([1, n_tokens])
            * np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
            deletion_matrix=np.zeros([1, n_tokens]),
            n_rows_paired=1,
            msa_mask=np.ones([1, n_tokens]),
            msa_profile=np.zeros([n_tokens, len(STANDARD_RESIDUES_WITH_GAP_1)]),
            deletion_mean=np.zeros(n_tokens),
        )

    # Mask token positions that correspond to padding
    token_starts_with_stop, _ = extract_starts_entities(atom_array)
    token_mask = np.zeros(n_tokens)
    token_mask[: len(token_starts_with_stop[:-1])] = 1
    msa_feature_precursor.msa_mask = (
        msa_feature_precursor.msa_mask * token_mask[np.newaxis, :]
    )

    return msa_feature_precursor
