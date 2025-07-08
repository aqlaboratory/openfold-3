"""This module contains building blocks for MSA processing."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict, deque
from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.resources.residues import MoleculeType

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
        metadata (pd.DataFrame | list | np.ndarray):
            A list of metadata parsed from sequence headers of the MSA. If not provided,
            an empty DataFrame is assigned."""

    msa: np.ndarray[str]
    deletion_matrix: np.ndarray[int]
    metadata: pd.DataFrame | list | np.ndarray = dataclasses.field(
        default_factory=pd.DataFrame
    )

    def __len__(self):
        return self.msa.shape[0]

    def truncate(
        self, row_slice: int | slice, inplace: bool = False
    ) -> None | MsaArray:
        """Truncate the MSA to a maximum number of sequences.

        Args:
            row_slice (int | slice):
                Number of sequences to keep in the MSA or a slice object applied along
                the first axis of the MSA numpy array/metadata DataFrame.
            inplace (bool, optional):
                Whether to perform the operation in place. Defaults to False.

        Returns:
            None
        """
        # Convert to slice
        if isinstance(row_slice, int):
            row_slice = slice(row_slice)
        elif not isinstance(row_slice, slice):
            ValueError(
                "Argument max_seq_count should be an integer or a slice."
                f"but got {type(row_slice)}."
            )

        # Make sure the slice is within the bounds
        row_slice = slice(min(row_slice.stop, self.__len__()))

        # Truncate
        if inplace:
            self.msa = self.msa[row_slice, :]
            self.deletion_matrix = self.deletion_matrix[row_slice, :]
            if isinstance(self.metadata, pd.DataFrame):
                self.metadata = self.metadata.iloc[row_slice]
            else:
                self.metadata = self.metadata[row_slice]
            return None
        else:
            return MsaArray(
                msa=self.msa[row_slice, :],
                deletion_matrix=self.deletion_matrix[row_slice, :],
                metadata=self.metadata.iloc[row_slice]
                if isinstance(self.metadata, pd.DataFrame)
                else self.metadata[row_slice],
            )

    def concatenate(
        self, msa_array: MsaArray, axis: int, inplace: bool = False
    ) -> MsaArray | None:
        """Concatenate the MsaArray with another MsaArray.

        Note: only keeps metadata if 1) concatenation is done vertically i.e. along axis
        0 and 2) if both metadata fields has type list or np.ndarray. Otherwise, does
        not keep metadata and replaces it with an empty DataFrame.

        Args:
            msa_array (MsaArray):
                The MsaArray object to concatenate with the current MsaArray.
            axis (int):
                The axis along which to concatenate the MSA arrays.
            inplace (bool, optional):
                Whether to perform the operation in place. Defaults to False.

        Returns:
            MsaArray | None:
                A new MsaArray object containing the concatenated MSA arrays or None if
                `inplace=True`.
        """
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 (rows) or 1 (columns), got {axis}.")

        # Extract arrays and metadata
        a1, a2 = self.msa, msa_array.msa
        d1, d2 = self.deletion_matrix, msa_array.deletion_matrix

        # Validate shapes and determine metadata concatenation function
        if axis == 0:
            if a1.shape[1] != a2.shape[1] or d1.shape[1] != d2.shape[1]:
                raise ValueError(
                    "Cannot concatenate along axis=0: number of columns must match "
                    f"(msa {a1.shape[1]} vs {a2.shape[1]}, "
                    f"deletion {d1.shape[1]} vs {d2.shape[1]})."
                )
            # Preserve metadata if both are list/ndarray
            if isinstance(self.metadata, (list, np.ndarray)) and isinstance(
                msa_array.metadata, (list, np.ndarray)
            ):
                metadata_concat_fn = partial(np.concatenate, axis=0)
            else:
                metadata_concat_fn = None
        else:
            if a1.shape[0] != a2.shape[0] or d1.shape[0] != d2.shape[0]:
                raise ValueError(
                    "Cannot concatenate along axis=1: number of rows must match "
                    f"(msa {a1.shape[0]} vs {a2.shape[0]}, "
                    f"deletion {d1.shape[0]} vs {d2.shape[0]})."
                )
            metadata_concat_fn = None
        array_concat_fn = partial(np.concatenate, axis=axis)

        # Perform concatenation
        msa_cat = array_concat_fn([a1, a2])
        del_cat = array_concat_fn([d1, d2])
        metadata_cat = (
            metadata_concat_fn(
                [np.asarray(self.metadata), np.asarray(msa_array.metadata)]
            )
            if metadata_concat_fn
            else pd.DataFrame()
        )

        if inplace:
            self.msa = msa_cat
            self.deletion_matrix = del_cat
            self.metadata = metadata_cat
            return None

        return MsaArray(msa=msa_cat, deletion_matrix=del_cat, metadata=metadata_cat)

    @classmethod
    def multi_concatenate(
        cls,
        msa_arrays: Sequence[MsaArray],
        axis: int = 0,
    ) -> MsaArray:
        """
        Concatenate a sequence of MsaArray objects.

        Args:
            msa_arrays (Sequence[MsaArray]):
                A sequence of MsaArray objects to concatenate.
            axis (int):
                The axis along which to concatenate the MSA arrays. 0 for rows, 1 for
                columns.

        Returns:
            A new MsaArray (unless `inplace=True`, then None).
        """
        if not msa_arrays:
            raise ValueError("Need at least one MsaArray to concatenate.")

        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")

        # Pull out the raw arrays
        msas = [i.msa for i in msa_arrays]
        del_matrices = [i.deletion_matrix for i in msa_arrays]
        metas = [i.metadata for i in msa_arrays]

        # Validate shapes
        if axis == 0:
            # all must have same number of columns
            ncols_msa = msas[0].shape[1]
            ncols_del = del_matrices[0].shape[1]
            if any(m.shape[1] != ncols_msa for m in msas):
                raise ValueError(
                    "All msas must have the same number of columns to concat on axis=0."
                )
            if any(d.shape[1] != ncols_del for d in del_matrices):
                raise ValueError(
                    "All deletion_matrices must have the same number of columns to "
                    "concat on axis=0."
                )
            # metadata: can only stitch if all are array-like
            if all(isinstance(md, pd.DataFrame) for md in metas):
                meta_concat = pd.DataFrame()  # pd.concat(metas, ignore_index=True)
            elif all(isinstance(md, (list, np.ndarray)) for md in metas):
                meta_concat = np.concatenate([np.asarray(md) for md in metas], axis=0)
            else:
                meta_concat = pd.DataFrame()
        else:
            # axis == 1: all must have same number of rows
            nrows_msa = msas[0].shape[0]
            nrows_del = del_matrices[0].shape[0]
            if any(m.shape[0] != nrows_msa for m in msas):
                raise ValueError(
                    "All msas must have the same number of rows to concat on axis=1."
                )
            if any(d.shape[0] != nrows_del for d in del_matrices):
                raise ValueError(
                    "All deletion_matrices must have the same number of rows to "
                    "concat on axis=1."
                )
            # never preserve metadata when concatenating columns
            meta_concat = pd.DataFrame()

        # Perform the concatenations
        msa_cat = np.concatenate(msas, axis=axis)
        del_cat = np.concatenate(del_matrices, axis=axis)

        return cls(
            msa=msa_cat,
            deletion_matrix=del_cat,
            metadata=meta_concat,
        )

    def pad(
        self,
        target_length: int,
        axis: int,
        pad_value: str = "-",
        return_mask: bool = True,
        inplace: bool = False,
    ) -> MsaArray | None | tuple[MsaArray | None, np.ndarray]:
        """Pad the MsaArray.

        Note: Only the msa and deletion matrix is padded, not the metadata. Padding when
        target_length is less than the array size is not supported. If the array
        dimension changes after padding, metadata is replaced with an empty DataFrame,
        otherwise kept the same.

        Args:
            target_length (int):
                The target length of the MSA array along the specified axis.
            axis (int):
                The axis along which to pad the MSA array.
            pad_value (str, optional):
                The value to use for padding the msa. Defaults to "-". The deletion
                matrix is always padded with 0s.
            return_mask (bool, optional):
                Whether to return a mask indicating the padded regions. Defaults to
                True.
            inplace (bool, optional):
                Whether to perform the operation in place. Defaults to False.

        Returns:
            "MsaArray" | None | tuple["MsaArray" | None, np.ndarray]:
                If inplace is True, returns None. Otherwise, returns a new MsaArray
                object with the padded MSA arrays. If return_mask is True, also returns
                a mask indicating the padded regions.
        """
        current_size = self.msa.shape[axis]
        pad_width = target_length - current_size

        # Error if padding is negative
        if pad_width < 0:
            raise ValueError(
                f"MsaArray of shape {self.msa.shape} cannot be padded to "
                f"a smaller size {target_length} along axis {axis}."
            )

        # Return unmodified array if no padding is needed
        elif pad_width == 0:
            if inplace:
                if return_mask:
                    return None, np.ones(self.msa.shape, dtype=int)
                else:
                    return None
            else:
                msa_array_padded = MsaArray(
                    msa=self.msa.copy(),
                    deletion_matrix=self.deletion_matrix.copy(),
                    metadata=self.metadata.copy(),
                )
                if return_mask:
                    mask = np.ones(self.msa.shape, dtype=int)
                    return msa_array_padded, mask
                else:
                    return msa_array_padded

        # Actually pad
        else:
            pad_widths = [(0, 0)] * self.msa.ndim
            pad_widths[axis] = (0, pad_width)
            if inplace:
                self.msa = np.pad(
                    self.msa,
                    pad_widths,
                    mode="constant",
                    constant_values=pad_value,
                )
                self.deletion_matrix = np.pad(
                    self.deletion_matrix,
                    pad_widths,
                    mode="constant",
                    constant_values=0,
                )
                self.metadata = pd.DataFrame()
                if return_mask:
                    mask = self._make_padding_mask(self.msa, axis, current_size)
                    return None, mask
                else:
                    return None

            else:
                padded_msa = np.pad(
                    self.msa,
                    pad_widths,
                    mode="constant",
                    constant_values=pad_value,
                )
                padded_deletion_matrix = np.pad(
                    self.deletion_matrix,
                    pad_widths,
                    mode="constant",
                    constant_values=0,
                )
                msa_array_padded = MsaArray(
                    msa=padded_msa,
                    deletion_matrix=padded_deletion_matrix,
                    metadata=pd.DataFrame(),
                )
                if return_mask:
                    mask = self._make_padding_mask(padded_msa, axis, current_size)
                    return msa_array_padded, mask
                else:
                    return msa_array_padded

    def _make_padding_mask(
        self, padded_msa: np.ndarray, axis: int, current_size: int
    ) -> np.ndarray:
        """Makes mask for the padded MSA.

        Args:
            padded_msa (np.ndarray):
                The padded MSA array to be masked.
            axis (int):
                The axis along which the padding is applied.
            current_size (int):
                The input size of the array along the specified axis.

        Returns:
            np.ndarray:
                The padding mask.
        """
        # Init mask
        mask = np.ones(padded_msa.shape, dtype=int)
        # Create slice that selects full array
        indexer = [slice(None)] * padded_msa.ndim
        # Replace with slice for padded region from the unpadded size to the end of the
        # padding and replace with 0s
        indexer[axis] = slice(current_size, None)
        mask[tuple(indexer)] = 0
        return mask

    def to_dict(self) -> dict[np.ndarray]:
        """Casts the MsaArray attributes into a dict.

        Returns:
            dict[np.ndarray]:
                A dictionary containing the MsaArray attributes as numpy arrays.
        """
        return {
            "msa": self.msa,
            "deletion_matrix": self.deletion_matrix,
            "metadata": np.array(self.metadata),
        }

    def subset(self, row_mask: np.ndarray[bool]) -> MsaArray:
        """Subsets the MsaArray based on a boolean mask.

        Args:
            row_mask (np.ndarray[bool]):
                Boolean mask to subset the MSA array.

        Returns:
            MsaArray:
                A new MsaArray object containing the subsetted MSA arrays.
        """
        msa = self.msa[row_mask, :]
        deletion_matrix = self.deletion_matrix[row_mask, :]
        if isinstance(self.metadata, pd.DataFrame):
            metadata = self.metadata[row_mask]
        else:
            ## is python list
            metadata = [
                self.metadata[i] for i in range(len(self.metadata)) if row_mask[i]
            ]

        return MsaArray(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)


@dataclasses.dataclass(frozen=False)
class MsaArrayCollection:
    """Class representing a collection MSAs for a single sample.

    This class can be in one of three states:
        1.) parsed state:
            attributes rep_id_to_query_seq, rep_id_to_paired_msa and rep_id_to_main_msa
            populated after parsing and chain_id_to_query_seq, chain_id_to_paired_msa,
            chain_id_to_main_msa unpopulated.
        2.) processed state:
            attributes rep_id_to_query_seq, rep_id_to_paired_msa and rep_id_to_main_msa
            unpopulated and chain_id_to_query_seq, chain_id_to_paired_msa,
            chain_id_to_main_msa populated after processing.
        3.) prefeaturized state:
            Processed state with the attribute row_counts also populated.

    Attributes _state, chain_id_to_rep_id, chain_id_to_mol_type are populated in both
    states.

    Attributes:
        _state (str):
            The state of the MsaArrayCollection object. Can be one of "init", "parsed",
            "processed", or "prefeaturized".
        rep_id_to_query_seq (dict[str, np.ndarray[str]]):
            Dictionary mapping representative chain IDs to numpy arrays of their
            corresponding query sequences.
        rep_id_to_paired_msa (dict[str, MsaArray] | None):
            Dictionary mapping representative chain IDs to Msa objects containing paired
            MSA data. Only used if precomputed paired MSAs are provided.
        rep_id_to_main_msa (dict[str, dict[str, MsaArray]]):
            Dictionary mapping representative chain IDs to dictionaries of Msa objects.
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
        row_counts (dict[str, int | dict[str, int]]):
            Dictionary containing the number of total number of rows in the horizontally
            concatenated MSA capped to max_rows, the number of paired MSA rows after
            cropping and the representative id to number of main MSA rows.
    """

    # Core attributes
    chain_id_to_rep_id: dict[str, str]
    chain_id_to_mol_type: dict[str, str]  # TODO convert to dict[str, MoleculeType]
    _state: str = "init"

    # State parsed attributes
    rep_id_to_query_seq: dict[str, np.ndarray[str]] = dataclasses.field(
        default_factory=dict
    )
    rep_id_to_paired_msa: dict[str, MsaArray] | None = dataclasses.field(
        default_factory=dict
    )
    rep_id_to_main_msa: dict[str, dict[str, MsaArray]] = dataclasses.field(
        default_factory=dict
    )

    # State processed attributes
    chain_id_to_query_seq: dict[str, MsaArray] = dataclasses.field(default_factory=dict)
    chain_id_to_paired_msa: dict[str, MsaArray] = dataclasses.field(
        default_factory=dict
    )
    chain_id_to_main_msa: dict[str, MsaArray] = dataclasses.field(default_factory=dict)

    # Prefeaturized attributes
    row_counts: dict[str, int | dict[str, int]] = dataclasses.field(
        default_factory=dict
    )

    def set_state_parsed(
        self, rep_id_to_query_seq, rep_id_to_paired_msa=None, rep_id_to_main_msa=None
    ):
        """Set the state to parsed."""
        self._state = "parsed"
        self.rep_id_to_query_seq = rep_id_to_query_seq
        self.rep_id_to_paired_msa = rep_id_to_paired_msa if rep_id_to_paired_msa else {}
        self.rep_id_to_main_msa = rep_id_to_main_msa if rep_id_to_main_msa else {}
        self.chain_id_to_query_seq = {}
        self.chain_id_to_paired_msa = {}
        self.chain_id_to_main_msa = {}

    def set_state_processed(
        self, chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa
    ):
        """Set the state to processed."""
        self._state = "processed"
        self.chain_id_to_query_seq = chain_id_to_query_seq
        self.chain_id_to_paired_msa = chain_id_to_paired_msa
        self.chain_id_to_main_msa = chain_id_to_main_msa
        self.rep_id_to_query_seq = {}
        self.rep_id_to_paired_msa = {}
        self.rep_id_to_main_msa = {}

    def set_state_prefeaturized(self, n_rows, n_rows_paired_cropped, n_rows_main):
        """Set the state to prefeaturized."""
        self._state = "prefeaturized"
        self.row_counts = {
            "n_rows": n_rows,
            "n_rows_paired_cropped": n_rows_paired_cropped,
            "n_rows_main": n_rows_main,
        }


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-homo-mono")
def find_monomer_homomer(msa_array_collection: MsaArrayCollection) -> bool:
    """Determines if the sample is a monomer or homomer.

    Args:
        msa_array_collection (MsaCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        bool: Whether the sample is a monomer or a full homomer.
    """
    # Extract chain IDs and representative chain IDs
    chain_id_to_rep_id = {
        chain_id: rep_id
        for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items()
        if msa_array_collection.chain_id_to_mol_type[chain_id] == MoleculeType.PROTEIN
    }
    chain_ids, representative_chain_ids = (
        list(chain_id_to_rep_id.keys()),
        list(set(chain_id_to_rep_id.values())),
    )

    return (len(chain_ids) == 1) | (
        (len(representative_chain_ids) == 1) & (len(chain_ids) > 1)
    )


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-query")
def create_query_seqs(msa_array_collection: MsaArrayCollection) -> dict[int, MsaArray]:
    """Extracts and expands the query sequences and deletion matrices.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[int, MsaArray]:
            Dict of MsaArray objects containing the query sequence and deletion matrix
            for each chain, indexed by chain id.
    """
    return {
        k: MsaArray(
            msa=msa_array_collection.rep_id_to_query_seq[v],
            deletion_matrix=np.zeros(
                msa_array_collection.rep_id_to_query_seq[v].shape, dtype=int
            ),
            metadata=pd.DataFrame(),
        )
        for (k, v) in msa_array_collection.chain_id_to_rep_id.items()
    }


def extract_alignments_to_pair(
    msa_array_collection: MsaArrayCollection, msas_to_pair: Sequence[str] | None = None
) -> dict[str, MsaArray]:
    """Fetches the MsaArrays to be used for pairing from the MsaCollection.

    This function does not return MsaArrays for chains that only contain the query i.e.
    single sequence MSAs.

    Args:
        msa_array_collection (MsaCollection):
            A collection of MsaArray objects and chain IDs for a single sample.
        msas_to_pair (Sequence[str] | None):
            Sequence of strings indicating which MSAs files have species information
            that can be used for online pairing.

    Returns:
        dict[str, MsaArray]:
            Dict mapping chain IDs to MsaArray objects containing UniProt MSAs.
    """
    # No paired MSAs if no MSAs to pair
    msa_arrays_to_pair = {}
    if msas_to_pair is None:
        return msa_arrays_to_pair

    # Only pair across protein MSAs
    protein_rep_ids = set(
        rep_id
        for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items()
        if msa_array_collection.chain_id_to_mol_type[chain_id] == MoleculeType.PROTEIN
    )
    rep_ids = msa_array_collection.rep_id_to_main_msa.keys()

    # Get pairable MSAs, exclude MSAs only with query
    for rep_id in rep_ids:
        # Skip non-proteins
        if rep_id not in protein_rep_ids:
            continue

        rep_msa_map_per_chain = msa_array_collection.rep_id_to_main_msa[rep_id]

        # Find all MSA arrays that can be paired and concatenate vertically
        msa_arrays_to_pair_i = []
        for msa_name in msas_to_pair:
            m = rep_msa_map_per_chain.get(msa_name)
            if m is not None:
                msa_arrays_to_pair_i.append(m)

        msa_arrays_to_pair_cat = MsaArray.multi_concatenate(
            msa_arrays=msa_arrays_to_pair_i,
            axis=0,
        )

        if msa_arrays_to_pair_cat is not None and len(msa_arrays_to_pair_cat) > 1:
            msa_arrays_to_pair[rep_id] = msa_arrays_to_pair_cat

    return msa_arrays_to_pair


def process_msa_pairing_metadata(metadata_raw: list[str]) -> pd.DataFrame:
    """Parses the species info from a list of fasta headers.

    The list of headers are converted into a DataFrame containing the species IDs in the
    same order as the sequences in the alignment for all but the query sequence. If the
    MsaArray only contains the query sequence, an empty DataFrame is returned.

    Args:
        metadata_raw (list[str]):
            The raw list of metadata strings from the MSA file. Should contain the query
            header in the first position and all subsequent positions should have the
            format tr|<UniProt ID>|<UniProt ID>_<species ID>/<start idx>-end idx, for
            example: tr|A0A1W9RZR3|A0A1W9RZR3_9BACT/19-121

    Returns:
        pd.DataFrame:
            A DataFrame containing the sorted species IDs for the aligned sequences,
            with column name species_id.
    """

    # Embed into DataFrame
    if len(metadata_raw) == 1:
        # Empty DataFrame for UniProt MSAs that only contain the query sequence
        metadata = pd.DataFrame()
    else:
        metadata = pd.DataFrame({"raw": metadata_raw[1:]})
        metadata = metadata["raw"].str.split(r"[|_/:-]", expand=True)
        metadata.columns = [
            "tr",
            "uniprot_id",
            "uniprot_id_copy",
            "species_id",
            "chain_start",
            "chain_end",
        ]
        metadata = metadata[["species_id"]]

    return metadata


def sort_msa_by_distance_to_query(msa_array: MsaArray) -> None:
    """Reorders the MSA array based on the distance to the query sequence.

    Reorders all class attributes of the MSA object.

    Args:
        msa (MsaAarray):
            The parsed MsaArray.

    Returns:
        None
    """
    _msa_array = msa_array.msa
    distance_to_query = np.sum(_msa_array == _msa_array[0, :], axis=-1) / float(
        sum(_msa_array[0, :] != "-")
    )
    sorting_indices = np.argsort(distance_to_query)[::-1]
    msa_array.msa = _msa_array[sorting_indices, :]
    msa_array.metadata = msa_array.metadata.iloc[sorting_indices[1:] - 1]
    msa_array.deletion_matrix = msa_array.deletion_matrix[sorting_indices, :]


def count_species_per_chain(
    msa_arrays_to_pair: dict[str, MsaArray],
) -> tuple[np.ndarray[np.int], list[str]]:
    """Counts the occurrences of sequences from species in each chain's UniProt MSA.

    Args:
        msa_arrays_to_pair (dict[str, MsaArray]):
            Dict mapping chain IDs to Msa objects containing MSA arrays to pair with
            their species information.

    Returns:
        tuple[np.ndarray[np.int32], list[str]]:
            The array of occurrence counts (number of chains x number of unique species)
            and the list species with at least one sequence among MSAs of all chains.
    """
    species = []
    for chain_id in msa_arrays_to_pair:
        species.extend(set(msa_arrays_to_pair[chain_id].metadata["species_id"]))
    species = np.array(sorted(set(species)))
    species_index = np.arange(len(species))
    species_index_map = {species[i]: i for i in species_index}

    # Get lists of species per chain
    species_index_per_chain = [
        np.array(
            msa_arrays_to_pair[chain_id]
            .metadata["species_id"]
            .apply(lambda x: species_index_map[x])
        )
        for chain_id in msa_arrays_to_pair
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
    count_array = np.zeros((len(msa_arrays_to_pair), len(unique_integers)), dtype=int)

    # Use np.add.at for unbuffered in-place addition
    np.add.at(count_array, (list_indices, inverse_indices), 1)

    return count_array, species


def get_pairing_masks(
    count_array: np.ndarray[int], pairing_mask_keys: list[str]
) -> np.ndarray[bool]:
    """Generates masks for the pairing process.

    Useful for excluding things like species that occur only in one chain (will not be
    pairable) or species that occur too frequently in the MSA of a single chain (as done
    in the AF2-Multimer pairing code:
    https://github.com/google-deepmind/alphafold/blob/main/alphafold/data/msa_pairing.py#L216).

    Args:
        count_array (np.ndarray[np.int]):
            The array of species occurrence counts per chain.
        mask_keys (list[str]):
            List of strings indicating which mask to add.

    Returns:
        np.ndarray[np.bool]:
            The union of all masks to apply during pairing.
    """
    pairing_masks = np.ones(count_array.shape[1], dtype=bool)

    if "shared_by_two" in pairing_mask_keys:
        # Find species that are shared by at least two chains
        pairing_masks = pairing_masks & (np.sum(count_array != 0, axis=0) > 1)

    if "less_than_600" in pairing_mask_keys:
        # Find species that occur not more than 600 times in any single chain
        pairing_masks = pairing_masks & (
            np.sum(count_array <= 600, axis=0) == count_array.shape[0]
        )

    return pairing_masks


def find_pairing_indices(
    count_array: np.ndarray[int],
    pairing_masks: np.ndarray[bool],
    max_rows_paired: int,
    min_chains_paired_partial: int,
) -> np.ndarray[int]:
    """The main function for finding indices that pair rows in the MSA arrays.

    This function follows the AF2-Multimer strategy for pairing rows of UniProt MSAs
    with the added functionality to allow for excluding all partially paired rows with
    less than a certain number of chains. Here, the AF2 strategy excludes only fully
    unpaired i.e. block-diagonal elements, so the lowest number of chains allowed to be
    be partially paired is 2.

    Args:
        count_array (np.ndarray[int]):
            The array of species occurrence counts per chain
        pairing_masks (np.ndarray[bool]):
            The union of all masks to apply during pairing.
        max_rows_paired (int):
            The maximum number of rows to pair.
        min_chains_paired_partial (int):
            The minimum allowed number of chains to partially pair. Can be at most
            the number of unique chains in the crop or assembly.

    Returns:
        np.ndarray[int]:
            A tuple of arrays containing the indices that pair rows in MSAs of an
            assembly across chains with -1 at partially paired positions.
    """
    # Apply filters
    count_array_filtered = count_array[:, pairing_masks]
    species_index_filtered = np.arange(count_array.shape[-1])[pairing_masks]

    # Iteratively subtract from the counts array the number of sequences shared by
    # exactly n chains
    paired_species_rows = []  # species indices in the MSA feature format
    n_rows = 0  # number of paired rows
    n_unique_chains = count_array.shape[0]
    min_chains_paired_partial_local = min(min_chains_paired_partial, n_unique_chains)

    # Iterate over number of chains from max to min_chains_paired_partial_local
    for n in np.arange(min_chains_paired_partial_local, n_unique_chains + 1)[::-1]:
        # Find which species are shared by exactly n chains
        is_in_n_chains = np.sum(count_array_filtered > 0, axis=0) == n
        # skip if no species are shared by exactly n chains in the filtered array
        if sum(is_in_n_chains) == 0:
            continue

        # Find the lowest nonzero number of sequences across chains from shared species
        count_array_in_n_chains = count_array_filtered[:, is_in_n_chains]
        k_in_n_chains = np.where(
            count_array_in_n_chains > 0, count_array_in_n_chains, np.nan
        )
        min_in_n_chains = np.nanmin(k_in_n_chains, axis=0).astype(int)

        # Subset species indices to those shared by n chains
        species_in_n_chains = species_index_filtered[is_in_n_chains]

        # Expand filtered species index across occurrences and chains
        col_expand = np.repeat(species_in_n_chains, min_in_n_chains)
        chain_col_expand = np.tile(col_expand, (n_unique_chains, 1))
        n_rows += chain_col_expand.shape[1]

        # Mask missing species and add to chain list
        np.copyto(
            chain_col_expand,
            -1,
            where=np.repeat(count_array_in_n_chains, min_in_n_chains, axis=1) == 0,
        )
        paired_species_rows.append(chain_col_expand.T)

        # Subtract min per row for shared species at non-zero values
        paired_cols = count_array_filtered[:, is_in_n_chains].copy()
        count_array_filtered[:, is_in_n_chains] = np.where(
            paired_cols != 0, paired_cols - min_in_n_chains, paired_cols
        )

        # If row cutoff reached, crop final arrays to the row cutoff and break
        if n_rows >= max_rows_paired:
            n_rows_final = max_rows_paired - sum(
                [rows.shape[0] for rows in paired_species_rows[:-1]]
            )
            # Ensure n_rows_final is not negative
            n_rows_final = max(n_rows_final, 0)
            paired_species_rows[-1] = paired_species_rows[-1][:n_rows_final, :]
            break

    # Concatenate all paired arrays into a single paired array
    paired_rows_index = np.concatenate(paired_species_rows, axis=0)

    return paired_rows_index


def _num_encode_species(
    species: np.ndarray[str], species_array: np.ndarray[str]
) -> np.ndarray[int]:
    """Creates a numerical encoding of species names.

    Args:
        species (np.ndarray[str]):
            The order of species based on which to create the numerical encoding.
        species_array (np.ndarray[str]):
            The array of species names to encode.

    Returns:
        np.ndarray[int]:
            The numerical encoding of the species names, containing positional indices
            of the species names in the species array.
    """

    # Sort the species names and get the sorted indices
    species_sort_order = np.argsort(species)
    sorted_species = species[species_sort_order]

    # Assume species_array is your NumPy array of species names (strings)
    # Use searchsorted to find indices in the sorted array
    indices_in_sorted = np.searchsorted(sorted_species, species_array)

    # Verify matches to handle any possible mismatches
    matches = sorted_species[indices_in_sorted] == species_array

    # Initialize an array to hold the indices, defaulting to -1 for non-matches
    row_to_species = np.full(species_array.shape, -1, dtype=int)

    # Map the indices back to the original order
    row_to_species[matches] = species_sort_order[indices_in_sorted[matches]]

    return row_to_species


def map_to_paired_msa_per_chain(
    msa_array_collection: MsaArrayCollection,
    msa_arrays_to_pair: dict[str, MsaArray],
    paired_rows_index: np.ndarray[int],
    species: np.ndarray[str],
    mode: str = "deque",
) -> dict[str, MsaArray]:
    """Maps paired species indices to MSA rows i.e. seqences.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.
        msa_arrays_to_pair (dict[str, MsaArray]):
            Dict mapping chain IDs to Msa objects containing MSA arrays to pair with
            their species information.
        paired_rows_index (np.ndarray[int]):
            Array containing the indices that pair rows in MSAs of a crop/assembly
            across chains.
        species (np.ndarray[str]):
            Array of species with at least one sequence among MSAs of all chains, in the
            order used by entries of paired_rows_index.
        mode (str, optional):
            The mode to use for mapping paired species indices to MSA row indices.
            Defaults to "deque". Must be either "deque" or "outer_product". Generally
            should be set to deque as the outer_product mode requires the instantiation
            of large intermediate numpy arrays.

    Returns:
        dict[str, MsaArray]:
            Dict mapping chain IDs to MsaArrays containing the paired MSAs and paired
            deletion matrices. Metadata fields are empty.
    """

    # Map species indices back to MSA row indices
    # Pre-allocate MSA objects, including those without pairable MSAs
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
    for chain_idx, (chain_id, msa_array) in enumerate(msa_arrays_to_pair.items()):
        # Get the array of species for each aligned sequences to the query chain
        species_array = np.array(msa_array.metadata["species_id"].to_numpy())

        # Map to numerical using the species index
        row_to_species = _num_encode_species(species, species_array)

        # Get paired species ids for chain
        paired_row_index_of_chain = paired_rows_index[:, chain_idx]

        if mode == "deque":
            # Build a mapping from species index to deque of MSA row indices
            species_to_msa_rows = defaultdict(deque)
            for msa_row_idx, species_idx in enumerate(row_to_species):
                # Add 1 to msa_row_idx as the first rows is the query itself
                species_to_msa_rows[species_idx].append(msa_row_idx + 1)

            msa_rows = np.full(paired_row_index_of_chain.shape[0], -1, dtype=int)
            for i, species_idx in enumerate(paired_row_index_of_chain):
                if species_idx != -1:
                    msa_row_deque = species_to_msa_rows.get(species_idx)
                    if msa_row_deque and len(msa_row_deque) > 0:
                        msa_rows[i] = msa_row_deque.popleft()

        elif mode == "outer_product":
            # For each paired species row for the chain, find the position of
            # msa rows with the same species
            species_matches = row_to_species == paired_row_index_of_chain[:, np.newaxis]

            # Find MSA row indices for each paired species row
            used_rows = set()
            msa_rows = np.full(paired_row_index_of_chain.shape[0], -1, dtype=int)
            for row_idx, row in enumerate(species_matches):
                # Skip if the paired rows whose species index doesn't match any species
                # indices in the MSA: these are the masked rows
                if row.any():
                    species_matches_row = np.where(row)[0]
                    unused_rows_mask = ~np.isin(species_matches_row, list(used_rows))
                    if np.any(unused_rows_mask):
                        # Get first MSA row that hasn't been used yet
                        first_true_row = species_matches_row[
                            np.argmax(unused_rows_mask)
                        ]
                        # add 1 because the 1st row is the query itself
                        msa_rows[row_idx] = first_true_row + 1
                        used_rows.add(first_true_row)

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'deque' or 'outer_product'."
            )

        valid_rows = msa_rows != -1
        # Update MSA and deletion matrix with paired data
        paired_msa_per_chain[chain_id].msa[paired_row_index_of_chain != -1] = (
            msa_array.msa[msa_rows[valid_rows]]
        )
        paired_msa_per_chain[chain_id].deletion_matrix[
            paired_row_index_of_chain != -1
        ] = msa_array.deletion_matrix[msa_rows[valid_rows]]

    return paired_msa_per_chain


def expand_paired_msas(msa_array_collection: MsaArrayCollection) -> dict[int, MsaArray]:
    """Expands the paired msas and deletion matrices from representatives to chains.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[int, MsaArray]:
            Dict of MsaArray objects containing the paired msas and deletion matrix
            for each chain, indexed by chain id.
    """
    return {
        k: msa_array_collection.rep_id_to_paired_msa[v]
        for (k, v) in msa_array_collection.chain_id_to_rep_id.items()
    }


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-paired")
def create_paired(
    msa_array_collection: MsaArrayCollection,
    max_rows_paired: int,
    min_chains_paired_partial: int,
    pairing_mask_keys: list[str],
    msas_to_pair: Sequence[str] | None,
) -> dict[str, MsaArray]:
    """Creates paired MSA arrays from UniProt MSAs.

    Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on species
    IDs and sequence similarity to the query sequence with added functionality to
    exclude all partially paired rows with less than a certain number of chains
    as suggested by the AF3 SI.

    Also crops the paired MSA along its rows to max_rows_paired.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.
        max_rows_paired (int):
            The maximum number of rows to keep from the paired rows.
        min_chains_paired_partial (int):
            The minimum allowed number of chains to partially pair. Can be at most
            the number of unique chains in the crop or assembly.
        pairing_mask_keys (list[str]):
            List of strings indicating which mask to add.
        msas_to_pair (list[str]):
            Msas to to pair for online pairing
        paired_msa_order: (list[str])
            Order in which to pair msas

    Returns:
        dict[str, Msa]:
            Paired MSAs and deletion matrices for each chain.
    """
    # Get parsed uniprot hits
    msa_arrays_to_pair = extract_alignments_to_pair(msa_array_collection, msas_to_pair)

    # Ensure there are at least two chains with UniProt hits after filtering
    if len(msa_arrays_to_pair) <= 1:
        return {}

    # Process uniprot headers and sort by distance to query
    for chain_id in msa_arrays_to_pair:
        msa_arrays_to_pair[chain_id].metadata = process_msa_pairing_metadata(
            msa_arrays_to_pair[chain_id].metadata
        )
        sort_msa_by_distance_to_query(msa_arrays_to_pair[chain_id])

    # Count species occurrences per chain
    count_array, species = count_species_per_chain(msa_arrays_to_pair)

    # Get pairing masks
    pairing_masks = get_pairing_masks(count_array, pairing_mask_keys)

    # No valid pairs, skip MSA pairing
    if not np.any(pairing_masks):
        return {}

    # Find species indices that pair rows
    paired_rows_index = find_pairing_indices(
        count_array,
        pairing_masks,
        max_rows_paired,
        min_chains_paired_partial,
    )

    # Map species indices back to MSA row indices
    paired_msa_per_chain = map_to_paired_msa_per_chain(
        msa_array_collection,
        msa_arrays_to_pair,
        paired_rows_index,
        species,
    )

    # Expand paired MSAs across all chains
    msa_array_collection.rep_id_to_paired_msa = paired_msa_per_chain
    chain_id_to_paired_msa = expand_paired_msas(msa_array_collection)

    return chain_id_to_paired_msa


# TODO improve integration with existing create paired function
def create_paired_from_preprocessed(
    msa_array_collection: MsaArrayCollection,
    max_rows_paired: int,
    paired_msa_order: list[str],
) -> dict[str, MsaArray]:
    """Creates per-chain paired MSA arrays in the expected format from precomputed
    paired MSAs.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.
        max_rows_paired (int):
            The maximum number of rows to keep from the paired rows.
        paired_msa_order (list[str]):
            The order in which to concatenate the paired MSA arrays vertically if
            multiple are provided. Alignments not in this list are not added to the
            paired MSA stack.

    Returns:
        dict[str, MsaArray]: _description_
    """

    # Process precomputed paired MSAs
    processed_prepaired_msas = {}
    for rep_id, paired_msa_dict in msa_array_collection.rep_id_to_paired_msa.items():
        # Flatten
        prepaired_msa = MsaArray.multi_concatenate(
            [
                paired_msa_dict[paired_msa_key]
                for paired_msa_key in paired_msa_order
                if paired_msa_key in paired_msa_dict
            ]
        )
        # Crop
        processed_prepaired_msas[rep_id] = prepaired_msa.truncate(max_rows_paired)

    msa_array_collection.rep_id_to_paired_msa = processed_prepaired_msas

    # Map to per-chain
    return expand_paired_msas(msa_array_collection=msa_array_collection)


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-main")
def create_main(
    msa_array_collection: MsaArrayCollection,
    chain_id_to_paired_msa: dict[str, MsaArray],
    aln_order: list[str],
) -> dict[str, MsaArray]:
    """Creates main MSA arrays from non-UniProt MSAs.

    Note: this function also removes all sequences from the final main MSA that are
    present in the cropped paired MSA of the corresponding chain.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of MsaArrays and chain IDs for a single sample.
        chain_id_to_paired_msa (dict[str, MsaArray]):
            Dict of paired Msa objects per chain.
        aln_order (list[str]):
            The order in which to concatenate the main MSA arrays vertically.
            Alignments not in this list are not added to the main MSA.

    Returns:
        dict[str, MsaArray]:
            List of MsaArrays containing the main MSA arrays and deletion matrices
            for each chain.
    """
    # Iterate over representatives
    rep_main_msas = {}
    for rep_id, chain_data in msa_array_collection.rep_id_to_main_msa.items():
        chain_data = msa_array_collection.rep_id_to_main_msa[rep_id]

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
        if len(chain_id_to_paired_msa) > 0:
            # Create reprepsentative to chain ID mapping
            seen = set()
            rep_id_to_chain_id = {
                value: key
                for key, value in msa_array_collection.chain_id_to_rep_id.items()
                if value not in seen and not seen.add(value)
            }

            # The relevant paired MSA for this representative
            paired_arr = chain_id_to_paired_msa[rep_id_to_chain_id[rep_id]].msa
            arr = main_msa_redundant

            # 1) Convert each 2D array into a 1D "structured" view of type void This
            # way, each row is treated as one item.
            arr_view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
            paired_view = paired_arr.view(
                np.dtype((np.void, paired_arr.dtype.itemsize * paired_arr.shape[1]))
            )

            # 2) Vectorized membership check: is row in paired_msa? ~np.isin(...)
            # inverts the boolean array, so True -> "unique" row
            is_unique = np.squeeze(~np.isin(arr_view, paired_view), axis=-1)

            # Apply filtering with the boolean mask
            filtered_msa = main_msa_redundant[is_unique, :]
            filtered_deletion = main_deletion_matrix_redundant[is_unique, :]
        else:
            filtered_msa = main_msa_redundant
            filtered_deletion = main_deletion_matrix_redundant

        rep_main_msas[rep_id] = MsaArray(
            msa=filtered_msa,
            deletion_matrix=filtered_deletion,
            metadata=pd.DataFrame(),
        )

    # Reindex dicts from representatives to chain IDs
    main_msas = {}
    for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items():
        main_msas[chain_id] = rep_main_msas[rep_id]

    return main_msas
