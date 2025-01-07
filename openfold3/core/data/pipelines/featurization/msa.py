"""This module contains featurization pipelines for MSAs."""

import torch
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.msa import (
    create_msa_feature_precursor_af3,
)
from openfold3.core.data.primitives.featurization.structure import (
    encode_one_hot,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import MsaArrayCollection
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    get_with_unknown_1_to_idx,
)


@log_runtime_memory(runtime_dict_key="runtime-msa-feat")
def featurize_msa_af3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
    max_rows: int,
    max_rows_paired: int,
    n_tokens: int,
    subsample_with_bands: bool,
) -> dict[str, torch.Tensor]:
    """_summary_

    Args:
        atom_array (AtomArray):
            Target structure atom array.
        msa_array_collection (MsaArrayCollection):
            Collection of processed MSA arrays.
        max_rows (int):
            Maximum number of rows allowed in the MSA.
        max_rows_paired (int):
            Maximum number of paired rows allowed in the MSA.
        n_tokens (int):
            Number of tokens in the target structure.
        subsample_with_bands (bool):
            Whether to subsample the main MSA. Not currently implemented.

    Raises:
        NotImplementedError:
            If subsample_with_bands is True.

    Returns:
        dict[str, torch.Tensor]:
            Dictionary of MSA features.
    """
    # Create MsaFeaturePrecursorAF3 <- MSA-to-token mapping and subsampling logic goes
    # here, so [:max_rows, :] should be removed from below
    msa_feature_precursor = create_msa_feature_precursor_af3(
        atom_array=atom_array,
        msa_array_collection=msa_array_collection,
        max_rows=max_rows,
        max_rows_paired=max_rows_paired,
        token_budget=n_tokens,
    )

    if subsample_with_bands:
        raise NotImplementedError("Subsampling with bands is not implemented yet.")

    # Create features
    features = {}
    msa_restype_index = torch.tensor(
        get_with_unknown_1_to_idx(msa_feature_precursor.msa), dtype=torch.int64
    )
    features["msa"] = encode_one_hot(
        msa_restype_index, len(STANDARD_RESIDUES_WITH_GAP_1)
    ).to(torch.int32)
    deletion_matrix = torch.tensor(
        msa_feature_precursor.deletion_matrix, dtype=torch.int64
    )
    features["has_deletion"] = (deletion_matrix != 0).to(torch.float32)
    features["deletion_value"] = torch.atan(deletion_matrix / 3.0) * (
        2.0 / torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
    ).to(torch.float32)
    features["deletion_mean"] = torch.tensor(
        msa_feature_precursor.deletion_mean, dtype=torch.float32
    )
    features["profile"] = torch.tensor(
        msa_feature_precursor.msa_profile, dtype=torch.float32
    )

    features["num_paired_seqs"] = torch.tensor(
        [msa_feature_precursor.n_rows_paired], dtype=torch.int32
    )

    features["msa_mask"] = torch.tensor(
        msa_feature_precursor.msa_mask, dtype=torch.float32
    )

    return features
