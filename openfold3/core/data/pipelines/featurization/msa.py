# Copyright 2025 AlQuraishi Laboratory
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

"""This module contains featurization pipelines for MSAs."""

import torch
from biotite.structure import AtomArray
from pydantic import BaseModel

from openfold3.core.data.primitives.featurization.msa import (
    MsaFeaturePrecursorOF3,
    create_msa_feature_precursor_of3,
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
)


@log_runtime_memory(runtime_dict_key="runtime-msa-feat")
def featurize_msa_of3(
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
    # Create MsaFeaturePrecursorOF3
    msa_feature_precursor = create_msa_feature_precursor_of3(
        atom_array=atom_array,
        msa_array_collection=msa_array_collection,
        max_rows=max_rows,
        max_rows_paired=max_rows_paired,
        n_tokens=n_tokens,
    )

    if subsample_with_bands:
        raise NotImplementedError("Subsampling with bands is not implemented yet.")

    # Create features
    features = {}
    features["msa"] = encode_one_hot(
        torch.tensor(msa_feature_precursor.msa_index, dtype=torch.int64),
        len(STANDARD_RESIDUES_WITH_GAP_1),
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


class MsaFeaturizerOF3Config(BaseModel):
    max_rows: int
    max_rows_paired: int
    subsample_with_bands: bool


class MsaFeaturizerOF3:
    """Featurizer for MSAs."""

    def __init__(
        self,
        config: MsaFeaturizerOF3Config,
    ):
        self.max_rows = config.max_rows
        self.max_rows_paired = config.max_rows_paired
        self.subsample_with_bands = config.subsample_with_bands

    def create_feature_precursor(
        self,
        atom_array: AtomArray,
        msa_array_collection: MsaArrayCollection,
        n_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Create feature precursor for MSAs.

        Args:
            atom_array (AtomArray):
                Target structure atom array.
            msa_array_collection (MsaArrayCollection):
                Collection of processed MSA arrays.
            n_tokens (int):
                Number of tokens in the target structure.

        Returns:
            dict[str, torch.Tensor]:
                Dictionary of MSA features.
        """
        return create_msa_feature_precursor_of3(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            max_rows=self.max_rows,
            max_rows_paired=self.max_rows_paired,
            n_tokens=n_tokens,
        )

    def create_features(
        self, msa_feature_precursor: MsaFeaturePrecursorOF3
    ) -> dict[str, torch.Tensor]:
        """Create features from MSA feature precursor.

        Args:
            msa_feature_precursor (MsaFeaturePrecursorOF3):
                MSA feature precursor.

        Returns:
            dict[str, torch.Tensor]:
                Dictionary of MSA features.
        """

        if self.subsample_with_bands:
            raise NotImplementedError("Subsampling with bands is not implemented yet.")

        features = {}
        features["msa"] = encode_one_hot(
            torch.tensor(msa_feature_precursor.msa_index, dtype=torch.int64),
            len(STANDARD_RESIDUES_WITH_GAP_1),
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

    def __call__(
        self,
        atom_array: AtomArray,
        msa_array_collection: MsaArrayCollection,
        n_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Featurize MSAs.

        Args:
            atom_array (AtomArray):
                Target structure atom array.
            msa_array_collection (MsaArrayCollection):
                Collection of processed MSA arrays.
            n_tokens (int):
                Number of tokens in the target structure.

        Returns:
            dict[str, torch.Tensor]:
                Dictionary of MSA features.
        """
        msa_feature_precursor = self.create_feature_precursor(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            n_tokens=n_tokens,
        )
        return self.create_features(msa_feature_precursor)
