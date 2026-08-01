from __future__ import annotations

import pytest
import torch

from openfold3.core.kernels.triton.smooth_lddt_ball_query import (
    _query_with_pred,
    ball_query_smooth_lddt_loss,
    is_ball_query_triton_available,
    is_ball_query_triton_installed,
)

requires_triton_cuda = pytest.mark.skipif(
    not is_ball_query_triton_available(),
    reason="Triton ball-query smooth lDDT requires Triton and CUDA",
)


def _inputs(
    n_atom: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    x_gt = torch.randn((2, n_atom, 3), device="cuda", dtype=torch.float32) * 2.0
    atom_mask = torch.ones((2, n_atom), device="cuda")
    is_nucleotide = torch.zeros_like(atom_mask)
    is_nucleotide[1] = 1
    loss_atom_mask = atom_mask.clone()
    return x_gt, atom_mask, is_nucleotide, loss_atom_mask


def _dense_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    radius_protein: float = 15.0,
    radius_nucleotide: float = 30.0,
) -> torch.Tensor:
    pred_delta = x[..., :, None, :] - x[..., None, :, :]
    gt_delta = x_gt[..., :, None, :] - x_gt[..., None, :, :]
    d_pred = torch.sqrt(eps + pred_delta.float().square().sum(dim=-1))
    d_gt = torch.sqrt(eps + gt_delta.square().sum(dim=-1))
    distance_error = torch.abs(d_gt - d_pred)
    score = 0.25 * (
        torch.sigmoid(0.5 - distance_error)
        + torch.sigmoid(1.0 - distance_error)
        + torch.sigmoid(2.0 - distance_error)
        + torch.sigmoid(4.0 - distance_error)
    )

    radius = torch.where(
        is_nucleotide.bool(),
        torch.tensor(radius_nucleotide, device=x.device),
        torch.tensor(radius_protein, device=x.device),
    )
    valid_atom = atom_mask.bool() & loss_atom_mask.bool()
    non_self = ~torch.eye(x.shape[-2], device=x.device, dtype=torch.bool)
    pair_mask = valid_atom[..., :, None] & valid_atom[..., None, :] & non_self
    in_ball = d_gt < radius[..., :, None]
    mask_sum = pair_mask.sum(dim=(-1, -2)) + eps
    score_mean = (score * in_ball * pair_mask).sum(dim=(-1, -2)) / mask_sum
    in_ball_mean = (in_ball * pair_mask).sum(dim=(-1, -2)) / mask_sum
    return 1 - score_mean / (in_ball_mean + eps)


def test_availability_flags_are_consistent():
    installed = is_ball_query_triton_installed()
    available = is_ball_query_triton_available()

    assert isinstance(installed, bool)
    assert isinstance(available, bool)
    assert available == (installed and torch.cuda.is_available())


def test_unavailable_backend_has_clear_error():
    if is_ball_query_triton_available():
        pytest.skip("Backend is available")

    x = torch.zeros((1, 2, 3))
    mask = torch.ones((1, 2))
    with pytest.raises(RuntimeError, match="requires Triton and a CUDA device"):
        ball_query_smooth_lddt_loss(
            x=x,
            x_gt=x,
            atom_mask_gt=mask,
            is_nucleotide=mask,
            loss_atom_mask=mask,
            eps=1e-8,
            top_k=1,
        )


@requires_triton_cuda
def test_seed_zero_advances_and_reservoir_excludes_self():
    n_atom = 32
    top_k = 8
    coords = torch.zeros((1, n_atom, 3), device="cuda")
    lengths = torch.tensor([n_atom], device="cuda")

    idx, _, _ = _query_with_pred(
        p1=coords,
        p2=coords,
        pred=coords,
        is_nucleotide=torch.zeros((1, n_atom), device="cuda"),
        lengths1=lengths,
        lengths2=lengths,
        k=top_k,
        seed=0,
    )

    centers = torch.arange(n_atom, device="cuda").view(1, n_atom, 1)
    sorted_idx = torch.sort(idx, dim=-1).values
    assert idx.shape == (1, n_atom, top_k)
    assert (idx >= 0).all()
    assert not (idx == centers).any()
    assert (sorted_idx[..., 1:] != sorted_idx[..., :-1]).all()
    assert idx[0, 0].tolist() != [31, 2, 3, 4, 5, 6, 7, 8]


@requires_triton_cuda
@pytest.mark.parametrize("top_k", [None, 0, -1])
def test_rejects_invalid_top_k(top_k):
    x_gt, atom_mask, is_nucleotide, loss_atom_mask = _inputs(n_atom=4)

    with pytest.raises(ValueError, match="smooth_lddt_top_k"):
        ball_query_smooth_lddt_loss(
            x=x_gt,
            x_gt=x_gt,
            atom_mask_gt=atom_mask,
            is_nucleotide=is_nucleotide,
            loss_atom_mask=loss_atom_mask,
            eps=1e-8,
            top_k=top_k,
        )


@requires_triton_cuda
@pytest.mark.parametrize(
    ("radius_protein", "radius_nucleotide"),
    [(0.0, 30.0), (15.0, 0.0), (-1.0, 30.0)],
)
def test_rejects_nonpositive_radius(radius_protein, radius_nucleotide):
    x_gt, atom_mask, is_nucleotide, loss_atom_mask = _inputs(n_atom=4)

    with pytest.raises(ValueError, match="must be positive"):
        ball_query_smooth_lddt_loss(
            x=x_gt,
            x_gt=x_gt,
            atom_mask_gt=atom_mask,
            is_nucleotide=is_nucleotide,
            loss_atom_mask=loss_atom_mask,
            eps=1e-8,
            top_k=3,
            radius_protein=radius_protein,
            radius_nucleotide=radius_nucleotide,
        )


@requires_triton_cuda
def test_forward_and_backward_match_dense():
    eps = 1e-8
    x_gt, atom_mask, is_nucleotide, loss_atom_mask = _inputs()
    torch.manual_seed(11)
    noise = 0.1 * torch.randn_like(x_gt)
    x_triton = (x_gt + noise).requires_grad_(True)
    x_dense = (x_gt + noise).requires_grad_(True)

    triton_loss = ball_query_smooth_lddt_loss(
        x=x_triton,
        x_gt=x_gt,
        atom_mask_gt=atom_mask,
        is_nucleotide=is_nucleotide,
        loss_atom_mask=loss_atom_mask,
        eps=eps,
    )
    dense_loss = _dense_smooth_lddt_loss(
        x=x_dense,
        x_gt=x_gt,
        atom_mask=atom_mask,
        is_nucleotide=is_nucleotide,
        loss_atom_mask=loss_atom_mask,
        eps=eps,
    )
    triton_loss.sum().backward()
    dense_loss.sum().backward()

    torch.testing.assert_close(triton_loss, dense_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_triton.grad, x_dense.grad, atol=1e-3, rtol=5e-3)


@requires_triton_cuda
def test_batch_and_sample_dimensions_match_loop():
    torch.manual_seed(12)
    batch_size = 2
    n_sample = 3
    n_atom = 12
    x_gt = torch.randn((batch_size, n_atom, 3), device="cuda")
    atom_mask = torch.ones((batch_size, n_atom), device="cuda")
    is_nucleotide = torch.zeros_like(atom_mask)
    loss_atom_mask = atom_mask.clone()
    x_base = x_gt[:, None] + 0.1 * torch.randn(
        (batch_size, n_sample, n_atom, 3), device="cuda"
    )

    x_batched = x_base.detach().clone().requires_grad_(True)
    batched_loss = ball_query_smooth_lddt_loss(
        x=x_batched,
        x_gt=x_gt,
        atom_mask_gt=atom_mask,
        is_nucleotide=is_nucleotide,
        loss_atom_mask=loss_atom_mask,
        eps=1e-8,
        top_k=n_atom - 1,
        seed=5,
    )
    batched_loss.sum().backward()

    x_loop = x_base.detach().clone().requires_grad_(True)
    loop_loss = torch.stack(
        [
            torch.stack(
                [
                    ball_query_smooth_lddt_loss(
                        x=x_loop[b : b + 1, s],
                        x_gt=x_gt[b : b + 1],
                        atom_mask_gt=atom_mask[b : b + 1],
                        is_nucleotide=is_nucleotide[b : b + 1],
                        loss_atom_mask=loss_atom_mask[b : b + 1],
                        eps=1e-8,
                        top_k=n_atom - 1,
                        seed=5,
                    ).squeeze(0)
                    for s in range(n_sample)
                ]
            )
            for b in range(batch_size)
        ]
    )
    loop_loss.sum().backward()

    assert batched_loss.shape == (batch_size, n_sample)
    torch.testing.assert_close(batched_loss, loop_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_batched.grad, x_loop.grad, atol=1e-3, rtol=5e-3)


@requires_triton_cuda
def test_subsampling_is_repeatable_for_seed():
    x_gt, atom_mask, is_nucleotide, loss_atom_mask = _inputs()
    torch.manual_seed(13)
    x = x_gt + 0.2 * torch.randn_like(x_gt)

    kwargs = {
        "x": x,
        "x_gt": x_gt,
        "atom_mask_gt": atom_mask,
        "is_nucleotide": is_nucleotide,
        "loss_atom_mask": loss_atom_mask,
        "eps": 1e-8,
        "top_k": 3,
        "seed": 123,
    }
    first = ball_query_smooth_lddt_loss(**kwargs)
    second = ball_query_smooth_lddt_loss(**kwargs)

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@requires_triton_cuda
def test_bfloat16_backward_is_finite():
    x_gt, atom_mask, is_nucleotide, loss_atom_mask = _inputs()
    x = (x_gt + 0.05 * torch.randn_like(x_gt)).bfloat16().requires_grad_(True)

    loss = ball_query_smooth_lddt_loss(
        x=x,
        x_gt=x_gt,
        atom_mask_gt=atom_mask,
        is_nucleotide=is_nucleotide,
        loss_atom_mask=loss_atom_mask,
        eps=1e-8,
        top_k=x_gt.shape[-2] - 1,
    )
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all()


@requires_triton_cuda
def test_adjustable_radius_changes_loss():
    n_atom = 8
    x_gt = torch.zeros((1, n_atom, 3), device="cuda")
    x_gt[0, :, 0] = torch.arange(n_atom, device="cuda") * 6.0
    atom_mask = torch.ones((1, n_atom), device="cuda")
    is_nucleotide = torch.zeros_like(atom_mask)
    x = x_gt + 0.05 * torch.randn_like(x_gt)

    kwargs = {
        "x": x,
        "x_gt": x_gt,
        "atom_mask_gt": atom_mask,
        "is_nucleotide": is_nucleotide,
        "loss_atom_mask": atom_mask,
        "eps": 1e-8,
        "top_k": n_atom - 1,
    }
    default_loss = ball_query_smooth_lddt_loss(**kwargs)
    tight_loss = ball_query_smooth_lddt_loss(
        **kwargs,
        radius_protein=5.0,
        radius_nucleotide=5.0,
    )

    assert torch.isfinite(default_loss).all()
    assert torch.isfinite(tight_loss).all()
    assert not torch.allclose(default_loss, tight_loss, atol=1e-6, rtol=1e-6)
