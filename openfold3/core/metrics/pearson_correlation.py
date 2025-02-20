# This code is a modified version of the original code from the PyTorch Lightning
# library that handles the computation of the Pearson correlation coefficient in a
# distributed setting when using multiple devices and when some of the devices may
# have zero samples. The original code is available at:
# https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/regression/pearson.py

import logging

from torch import Tensor
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression.pearson import (
    _pearson_corrcoef_compute,
)

logger = logging.getLogger(__name__)


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    covs_xy: Tensor,
    counts: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Aggregates per-device statistics (means, variances, and covariance) into global
    statistics using the parallel algorithm, filtering out devices with no samples.

    If no valid devices remain after filtering, a warning is issued and NaNs
    are returned.

    Args:
        means_x: 1-D tensor of x-means from each device.
        means_y: 1-D tensor of y-means from each device.
        vars_x: 1-D tensor of x-variances from each device.
        vars_y: 1-D tensor of y-variances from each device.
        covs_xy: 1-D tensor of x-y covariances from each device.
        counts: 1-D tensor of sample counts from each device.

    Returns:
        A tuple of aggregated statistics:
            (mean_x, mean_y, var_x, var_y, cov_xy, total_count)
    """
    # Filter out devices with zero samples.
    valid = counts > 0
    if not valid.any():
        return means_x[0], means_y[0], vars_x[0], vars_y[0], covs_xy[0], counts[0]

    means_x, means_y = means_x[valid], means_y[valid]
    vars_x, vars_y = vars_x[valid], vars_y[valid]
    covs_xy, counts = covs_xy[valid], counts[valid]

    if len(means_x) == 1:
        return means_x[0], means_y[0], vars_x[0], vars_y[0], covs_xy[0], counts[0]

    # Initialize aggregated statistics with the first valid device's stats.
    agg_mx, agg_my, agg_vx, agg_vy, agg_cov, agg_n = (
        means_x[0],
        means_y[0],
        vars_x[0],
        vars_y[0],
        covs_xy[0],
        counts[0],
    )

    for i in range(1, len(means_x)):
        curr_mx, curr_my, curr_vx, curr_vy, curr_cov, curr_n = (
            means_x[i],
            means_y[i],
            vars_x[i],
            vars_y[i],
            covs_xy[i],
            counts[i],
        )
        tot_n = agg_n + curr_n
        delta_mx, delta_my = curr_mx - agg_mx, curr_my - agg_my
        factor = agg_n * curr_n / tot_n

        # Update aggregated statistics.
        agg_mx, agg_my = (
            (agg_n * agg_mx + curr_n * curr_mx) / tot_n,
            (agg_n * agg_my + curr_n * curr_my) / tot_n,
        )
        agg_vx += curr_vx + factor * (delta_mx**2)
        agg_vy += curr_vy + factor * (delta_my**2)
        agg_cov += curr_cov + factor * delta_mx * delta_my
        agg_n = tot_n

    return agg_mx, agg_my, agg_vx, agg_vy, agg_cov, agg_n


class ZeroSafePearsonCorrCoef(PearsonCorrCoef):
    """A version of PearsonCorrCoef that is safe for zero containing tensors."""

    def compute(self) -> Tensor:
        """Compute pearson correlation coefficient over state."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (
            self.num_outputs > 1 and self.mean_x.ndim > 1
        ):
            # multiple devices, need further reduction
            _, _, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x,
                self.mean_y,
                self.var_x,
                self.var_y,
                self.corr_xy,
                self.n_total,
            )
        else:
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _pearson_corrcoef_compute(var_x, var_y, corr_xy, n_total)
