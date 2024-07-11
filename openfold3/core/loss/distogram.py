import torch
import torch.nn as nn
import torch.nn.functional as F


def distogram_loss(
    x_gt: torch.Tensor,
    frame_idx: torch.Tensor,
    token_mask: torch.Tensor,
    logits: torch.Tensor,
    n_bins: int = 64,
    bin_min: float = 2.0,
    bin_max: float = 22.0,
) -> torch.Tensor:
    """
    Distogram loss
    See 1.9.8 of the AF2 supplementary.
    Args:
        x_gt:
            Ground truth coordinates. [*, N, 3]
        frame_idx:
            Frame indices (see predictedAlignmentError for more details). [*, T, 3]
        token_mask:
            Padding for unused token. [*, T]
        logits:
            Predicted distance error logits. [*, T, T, n_bins]
        n_bins:
            Number of bins.
        bin_min:
            Minimum lower bin.
        bin_max:
            Maximum upper bin.
    Returns:
        distogram_loss:
    """

    # coordinates of center atoms for each token
    center_idx = frame_idx[..., 1:2]  # [*, T, 1]
    T = center_idx.shape[-2]

    # [*, T, 3]
    x_gt_token = torch.gather(
        x_gt.unsqueeze(-3).expand(-1, T, -1, -1),  # [*, T, N, 3]
        dim=-2,
        index=center_idx.unsqueeze(-1).expand(-1, -1, -1, 3),  # [*, T, 1, 3]
    ).squeeze(-2)

    d = torch.cdist(x_gt_token, x_gt_token)  # [*, T, T]

    # boundaries = [0.0, 2.3125, ...,21.6875, 22.0]
    boundaries = torch.linspace(
        bin_min,
        bin_max,
        n_bins + 1,
        device=logits.device,
    )

    bins = torch.bucketize(d, boundaries[1:-1])  # [*, T, T]

    # set bins to -100 (ignore_index) for masked token
    square_mask = token_mask[..., None] * token_mask[..., None, :]  # [*, T, T]
    bins = bins.masked_fill(~square_mask.bool(), -100)  # [*, T, T]

    distogram_loss = F.cross_entropy(logits, bins)

    return distogram_loss


class DistogramLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.loss.distogram

    def forward(self, batch, output):
        x_gt = batch["x_gt"]  # [*, N, 3]
        frame_idx = batch["frame_idx"]  # [*, T, 3]
        token_mask = batch["token_mask"]  # [*, T]
        distogram_logits = output["p_distogram"]  # [*, T, T, 64]

        return distogram_loss(
            x_gt,
            frame_idx,
            token_mask,
            distogram_logits,
            self.config.loss.n_bins,
            self.config.loss.bin_min,
            self.config.loss.bin_max,
        )
