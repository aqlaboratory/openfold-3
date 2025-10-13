"""This module contains padding primitives."""

import torch


def pad_token_dim(
    features: dict[str, torch.Tensor],
    token_budget: int,
    token_dim_index_map: dict[str, int],
    pad_value: int | float | None = 0,
) -> dict[str, torch.Tensor]:
    """Pads a dict of tensors along the token dimension to a given budget.

    Args:
        features (dict[str, torch.Tensor]):
            Dictionary of features to pad.
        token_budget (int):
            Desired token dimension size.
        token_dim_index_map (dict[str, int]):
            Mapping of feature names to the index of the token dimension.
        pad_value (Optional[Union[int, float]]):
            Value to use as padding value. Defaults to 0.

    Returns:
        dict[str, torch.Tensor]: _description_
    """
    for feature_name, token_dim in token_dim_index_map.items():
        if feature_name in features:
            feature = features[feature_name]
            dim_sizes = [dim_size for dim_size in feature.shape]
            dim_sizes_padded = [
                dim_size if (i not in token_dim) else token_budget
                for dim_size, i in zip(dim_sizes, range(-len(dim_sizes), 0))
            ]
            feature_padded = (
                torch.ones(dim_sizes_padded, dtype=feature.dtype, device=feature.device)
                * pad_value
            )
            feature_padded[
                tuple(
                    slice(start, stop)
                    for start, stop in zip([0] * len(dim_sizes), dim_sizes)
                )
            ] = feature
            features[feature_name] = feature_padded
    return features
