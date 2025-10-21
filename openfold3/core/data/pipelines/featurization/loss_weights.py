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

"""
Pipelines for setting the loss weights in the FeatureDict.
"""

import copy

import torch


def set_loss_weights(
    loss_settings: dict, resolution: float | None
) -> dict[str, torch.Tensor]:
    """Updates and tensorizes loss weights in the FeatureDict based on the resolution.

    Args:
        loss_settings (dict):
            Dictionary parsed from the dataset_config containing
                - confidence_loss_names
                - diffusion_loss_names
                - loss_weight
                - min_resolution
                - max_resolution
        resolution (float | None):
            The resolution of the input data.

    Returns:
        dict[str, torch.Tensor]:
            Dictionary containing the loss settings.
    """
    loss_weight = copy.deepcopy(loss_settings["loss_weights"])
    if (resolution is None) or (
        resolution < loss_settings["min_resolution"]
        or resolution > loss_settings["max_resolution"]
    ):
        # Set all confidence losses to 0
        for loss_name in loss_settings["confidence_loss_names"]:
            loss_weight[loss_name] = 0

    return {k: torch.tensor([v], dtype=torch.float32) for k, v in loss_weight.items()}


def set_loss_weights_for_disordered_set(
    loss_settings: dict,
    resolution: float,
    disable_non_protein_diffusion_weights: bool,
) -> dict[str, torch.Tensor]:
    """Updates and tensorizes loss weights in the FeatureDict based on the resolution.
    Includes settings specific to the disordered PDB dataset.

    Args:
        loss_settings (dict):
            Dictionary parsed from the dataset_config containing
                - confidence_loss_names
                - diffusion_loss_names
                - loss_weight
                - min_resolution
                - max_resolution
        resolution (float):
            The resolution of the input data.
        disable_non_protein_diffusion_weights (bool):
            Whether loss mode should disable diffusion weights for non-proteins

    Returns:
        dict[str, Any]:
            Dictionary containing the loss settings for the disordered PDB dataset.
    """
    loss_settings_dict = set_loss_weights(loss_settings, resolution)
    loss_settings_dict["disable_non_protein_diffusion_weights"] = torch.tensor(
        [disable_non_protein_diffusion_weights], dtype=torch.bool
    )
    return loss_settings_dict
