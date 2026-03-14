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

"""This module contains padding primitives."""

import torch

TOKEN_DIM_INDEX_MAP = {
    "residue_index": [-1],
    "token_index": [-1],
    "asym_id": [-1],
    "entity_id": [-1],
    "sym_id": [-1],
    "restype": [-2],
    "is_protein": [-1],
    "is_rna": [-1],
    "is_dna": [-1],
    "is_ligand": [-1],
    "token_bonds": [-1, -2],
    "num_atoms_per_token": [-1],
    "is_atomized": [-1],
    "start_atom_index": [-1],
    "token_mask": [-1],
    "mol_entity_id": [-1],
    "mol_sym_id": [-1],
    "mol_sym_token_index": [-1],
    "mol_sym_component_id": [-1],
}


def pad_token_dim(
    features: dict[str, torch.Tensor],
    token_budget: int,
    pad_value: int | float | None = 0,
) -> dict[str, torch.Tensor]:
    """Pads a dict of tensors along the token dimension to a given budget.

    Args:
        features (dict[str, torch.Tensor]):
            Dictionary of features to pad.
        token_budget (int):
            Desired token dimension size.
        pad_value (Optional[Union[int, float]]):
            Value to use as padding value. Defaults to 0.

    Returns:
        dict[str, torch.Tensor]: _description_
    """
    for feature_name, token_dim in TOKEN_DIM_INDEX_MAP.items():
        if feature_name in features:
            feature = features[feature_name]
            dim_sizes = [dim_size for dim_size in feature.shape]
            dim_sizes_padded = [
                dim_size if (i not in token_dim) else token_budget
                for dim_size, i in zip(
                    dim_sizes, range(-len(dim_sizes), 0), strict=True
                )
            ]
            feature_padded = (
                torch.ones(dim_sizes_padded, dtype=feature.dtype, device=feature.device)
                * pad_value
            )
            feature_padded[
                tuple(
                    slice(start, stop)
                    for start, stop in zip([0] * len(dim_sizes), dim_sizes, strict=True)
                )
            ] = feature
            features[feature_name] = feature_padded
    return features
