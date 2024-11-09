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
    get_with_unknown_1,
)


@log_runtime_memory(runtime_dict_key="runtime-msa-feat")
def featurize_msa_af3(
    atom_array: AtomArray,
    msa_array_collection: MsaArrayCollection,
    max_rows: int,
    max_rows_paired: int,
    token_budget: int,
    subsample_with_bands: bool,
) -> dict[str, torch.Tensor]:
    # Create MsaFeaturePrecursorAF3 <- MSA-to-token mapping and subsampling logic goes
    # here, so [:max_rows, :] should be removed from below
    msa_feature_precursor = create_msa_feature_precursor_af3(
        atom_array=atom_array,
        msa_array_collection=msa_array_collection,
        max_rows_paired=max_rows_paired,
        token_budget=token_budget,
    )

    if subsample_with_bands:
        raise NotImplementedError("Subsampling with bands is not implemented yet.")

    # Create features
    features = {}
    msa_restype_index = torch.tensor(
        get_with_unknown_1(msa_feature_precursor.msa[:max_rows, :]), dtype=torch.int64
    )
    features["msa"] = encode_one_hot(
        msa_restype_index, len(STANDARD_RESIDUES_WITH_GAP_1)
    ).to(torch.int32)
    deletion_matrix = torch.tensor(
        msa_feature_precursor.deletion_matrix[:max_rows, :], dtype=torch.int64
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
    features["num_recycles"] = torch.randint(0, 5, (1,)).to(torch.int32)
    features["num_paired_seqs"] = torch.tensor(
        [msa_feature_precursor.n_rows_paired], dtype=torch.int32
    )

    features["msa_mask"] = torch.tensor(
        msa_feature_precursor.msa_mask[:max_rows, :], dtype=torch.float32
    )

    return features
