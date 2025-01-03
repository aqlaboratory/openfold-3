"""
Pipelines for setting the loss weights in the FeatureDict.
"""

import copy

import torch


def set_loss_weights(loss_settings: dict, resolution: float | None) -> dict[str, float]:
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
        dict[str, float]: _description_
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
