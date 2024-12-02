"""This module contains featurization pipelines for MSAs."""

import torch

from openfold3.core.data.primitives.featurization.structure import (
    encode_one_hot,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    get_with_unknown_1,
)


@log_runtime_memory(runtime_dict_key="runtime-msa-feat")
def featurize_msa_af3(msa_processed):
    max_rows = 16384

    features = {}
    msa_restype_index = torch.tensor(
        get_with_unknown_1(msa_processed.msa[:max_rows, :]), dtype=torch.int64
    )
    features["msa"] = encode_one_hot(
        msa_restype_index, len(STANDARD_RESIDUES_WITH_GAP_1)
    ).to(torch.int32)
    deletion_matrix = torch.tensor(
        msa_processed.deletion_matrix[:max_rows, :], dtype=torch.int64
    )
    features["has_deletion"] = (deletion_matrix != 0).to(torch.float32)
    features["deletion_value"] = torch.atan(deletion_matrix / 3.0) * (
        2.0 / torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
    ).to(torch.float32)
    features["deletion_mean"] = torch.tensor(
        msa_processed.deletion_mean, dtype=torch.float32
    )
    features["profile"] = torch.tensor(msa_processed.msa_profile, dtype=torch.float32)

    features["num_paired_seqs"] = torch.tensor(
        [msa_processed.n_rows_paired], dtype=torch.int32
    )

    features["msa_mask"] = torch.tensor(
        msa_processed.msa_mask[:max_rows, :], dtype=torch.float32
    )

    return features
