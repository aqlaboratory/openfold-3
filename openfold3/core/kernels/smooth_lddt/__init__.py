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
#
# Modified from PyTorch3D's ball-query autograd layer
# (https://github.com/facebookresearch/pytorch3d, Meta Platforms, Inc.,
#  BSD-3-Clause; see
#  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
# by Liang Hong <lhong22@cse.cuhk.edu.hk>. The custom autograd Function
# (_BallQueryWithPredDist) follows PyTorch3D's _ball_query +
# knn_points_backward pattern: forward saves only (pred, idx) and
# backward dispatches to a dedicated CUDA kernel that scatters gradients
# via atomicAdd, so the [B, N, K, 3] x_j gather is never materialized.
# The checkpointed scoring stage and the bf16-native pred-distance path
# are OpenFold3-specific additions.

"""Smooth lDDT CUDA kernel helpers."""

import torch
import torch.utils.checkpoint

from openfold3.core.kernels.smooth_lddt.ball_query import (
    ball_query_coop_with_pred,
    ball_query_pred_backward,
    is_ball_query_available,
    is_ball_query_installed,
)

MAX_SMOOTH_LDDT_RADIUS = 30.0


def is_smooth_lddt_kernel_installed() -> bool:
    """Return whether the CUDA ball-query smooth lDDT backend can be built."""
    return is_ball_query_installed()


def is_smooth_lddt_kernel_available() -> bool:
    """Return whether the CUDA ball-query smooth lDDT backend is usable now."""
    return is_ball_query_available()


def _expand_to_x_batch(
    tensor: torch.Tensor, x: torch.Tensor, trailing_dims: int
) -> torch.Tensor:
    """Expand a batch feature tensor across any diffusion sample dimensions."""
    while tensor.ndim < len(x.shape[:-2]) + trailing_dims:
        tensor = tensor.unsqueeze(-trailing_dims - 1)

    return tensor.expand(*x.shape[:-2], *tensor.shape[-trailing_dims:])


def _flatten_batch(tensor: torch.Tensor, trailing_dims: int) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[-trailing_dims:])


COOP_W = 8


def _validate_top_k(top_k: int | None, n_atom: int, align_to_w: bool = False) -> int:
    if top_k is None:
        raise ValueError("smooth_lddt_top_k must be set for ball-query smooth lDDT")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("smooth_lddt_top_k must be a positive integer")

    k = min(top_k, max(n_atom - 1, 0))
    if align_to_w and k > 0:
        # Kernel receives (k+1); round that up to multiple of W
        k = ((k + COOP_W) // COOP_W) * COOP_W - 1
    return k


class _BallQueryWithPredDist(torch.autograd.Function):
    """Custom autograd Function: forward kernel emits dists_pred, backward uses
    a dedicated CUDA kernel that scatters gradients via atomicAdd over saved
    (pred, idx) — no ``x_j`` gather is ever materialized."""

    @staticmethod
    def forward(ctx, pred, p1_gt, p2_gt, lengths1, lengths2, k, radius, seed):
        idx, dists_gt, dists_pred = ball_query_coop_with_pred(
            p1=p1_gt,
            p2=p2_gt,
            pred=pred,
            lengths1=lengths1,
            lengths2=lengths2,
            k=k,
            radius=radius,
            seed=seed,
        )
        ctx.save_for_backward(pred, idx)
        ctx.mark_non_differentiable(idx, dists_gt)
        return dists_pred, dists_gt, idx

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_dists_pred, grad_dists_gt, grad_idx):
        pred, idx = ctx.saved_tensors
        # Backward kernel returns fp32 grad; cast to pred's dtype.
        grad_pred_fp32 = ball_query_pred_backward(pred, idx, grad_dists_pred)
        grad_pred = grad_pred_fp32.to(pred.dtype)
        return grad_pred, None, None, None, None, None, None, None


def _score_from_dists(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    valid_neighbor: torch.Tensor,
    non_self: torch.Tensor,
    loss_atom_mask_i: torch.Tensor,
    loss_atom_mask_j: torch.Tensor,
    flat_is_nucleotide: torch.Tensor,
    eps: float,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    """Pure scoring math: sqrt → abs → sigmoid → mask → reduce.

    All inputs are kernel/gather outputs (no further gathers). When wrapped
    in ``torch.utils.checkpoint``, every intermediate is freed between
    forward and backward.
    """
    dx = torch.sqrt(eps + dists_pred.clamp_min(0.0))
    dx_gt = torch.sqrt(eps + dists_gt.clamp_min(0.0))
    d = torch.abs(dx_gt - dx)
    e = 0.25 * (
        torch.sigmoid(0.5 - d)
        + torch.sigmoid(1.0 - d)
        + torch.sigmoid(2.0 - d)
        + torch.sigmoid(4.0 - d)
    )

    cutoff = torch.where(
        flat_is_nucleotide.bool(),
        torch.tensor(30.0, device=dx.device, dtype=dx_gt.dtype),
        torch.tensor(15.0, device=dx.device, dtype=dx_gt.dtype),
    )
    c = dx_gt < cutoff[..., None]

    mask = (valid_neighbor * non_self * loss_atom_mask_i * loss_atom_mask_j).bool()

    mask_sum = torch.sum(mask, dim=(-1, -2)) + eps
    ce_mean = torch.sum(c * e * mask, dim=(-1, -2)) / mask_sum
    c_mean = torch.sum(c * mask, dim=(-1, -2)) / mask_sum
    lddt = ce_mean / (c_mean + eps)

    return (1 - lddt).reshape(out_shape)


def ball_query_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    top_k: int | None,
    seed: int = 0,
) -> torch.Tensor:
    """
    Compute smooth lDDT using the CUDA ball-query kernel.

    Uses the warp-cooperative kernel with reservoir sampling for unbiased
    random neighbor selection when more than ``top_k`` atoms qualify within
    the radius. The ``seed`` parameter controls the RNG for reproducibility
    (e.g. pass the training step number).

    The result is exact relative to dense smooth lDDT when ``top_k`` covers
    every in-radius neighbor for each atom.

    Internal optimizations (forward kernel emits ``dists_pred`` directly +
    custom backward + checkpointed scoring) eliminate the ``x_j`` gather
    and all autograd-saved scoring intermediates. Backward uses ``atomicAdd``
    in fp32 and is therefore inherently nondeterministic.
    """
    if not x.is_cuda:
        raise RuntimeError("Ball-query smooth lDDT requires CUDA tensors")

    n_atom = x.shape[-2]
    k = _validate_top_k(top_k=top_k, n_atom=n_atom, align_to_w=True)
    if k == 0:
        return torch.ones(x.shape[:-2], device=x.device, dtype=x.dtype) + x.sum() * 0.0

    x_gt = _expand_to_x_batch(x_gt, x, trailing_dims=2)
    atom_mask_gt = _expand_to_x_batch(atom_mask_gt, x, trailing_dims=1)
    is_nucleotide = _expand_to_x_batch(is_nucleotide, x, trailing_dims=1)
    loss_atom_mask = _expand_to_x_batch(loss_atom_mask, x, trailing_dims=1)
    loss_atom_mask = loss_atom_mask * atom_mask_gt

    flat_x = _flatten_batch(x, trailing_dims=2)
    flat_x_gt = _flatten_batch(x_gt, trailing_dims=2)
    flat_atom_mask_gt = _flatten_batch(atom_mask_gt, trailing_dims=1)
    flat_is_nucleotide = _flatten_batch(is_nucleotide, trailing_dims=1)
    flat_loss_atom_mask = _flatten_batch(loss_atom_mask, trailing_dims=1)

    flat_batch = flat_x.shape[0]
    lengths = torch.full((flat_batch,), n_atom, device=x.device, dtype=torch.long)

    # Keep unresolved atoms from filling limited ball-query candidate slots.
    p2 = torch.where(
        flat_atom_mask_gt[..., None].bool(),
        flat_x_gt,
        torch.full_like(flat_x_gt, 1.0e6),
    )

    # Forward kernel emits (idx, dists_gt, dists_pred). Backward of dists_pred
    # uses the dedicated CUDA backward kernel (atomicAdd scatter).
    dists_pred, dists_gt, idx = _BallQueryWithPredDist.apply(
        flat_x,
        flat_x_gt,
        p2,
        lengths,
        lengths,
        k + 1,
        MAX_SMOOTH_LDDT_RADIUS,
        seed,
    )

    safe_idx = idx.clamp_min(0)
    flat_safe_idx = safe_idx.reshape(flat_batch, -1)

    # Mask building (no autograd saves: no gradient flows through these gathers).
    center_idx = torch.arange(n_atom, device=x.device).view(1, n_atom, 1)
    valid_neighbor = idx >= 0
    non_self = safe_idx != center_idx
    loss_atom_mask_i = flat_loss_atom_mask[..., None]
    loss_atom_mask_j = torch.gather(
        flat_loss_atom_mask, dim=1, index=flat_safe_idx
    ).reshape(flat_batch, n_atom, k + 1)

    # Wrap scoring in non-reentrant checkpoint to free all elementwise
    # intermediates (sqrt, abs, sigmoid, masked sums) between fwd and bwd.
    out_shape = x.shape[:-2]
    if dists_pred.requires_grad:
        return torch.utils.checkpoint.checkpoint(
            _score_from_dists,
            dists_pred,
            dists_gt,
            valid_neighbor,
            non_self,
            loss_atom_mask_i,
            loss_atom_mask_j,
            flat_is_nucleotide,
            eps,
            out_shape,
            use_reentrant=False,
        )
    return _score_from_dists(
        dists_pred,
        dists_gt,
        valid_neighbor,
        non_self,
        loss_atom_mask_i,
        loss_atom_mask_j,
        flat_is_nucleotide,
        eps,
        out_shape,
    )
