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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: length-generic Triton impl
# of ball-query smooth lDDT for fast and low memory loss calc.

"""Independent Triton ball-query smooth lDDT backend.

Call directly::

    from openfold3.core.kernels.triton.smooth_lddt_ball_query import (
        ball_query_smooth_lddt_loss,
    )
    loss = ball_query_smooth_lddt_loss(...)

Radii default to protein 15 Å / nucleotide 30 Å.
"""

from __future__ import annotations

import math

import torch
import torch.utils.checkpoint

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

# Length-invariant launch knobs (not autotuned per shape).
_FWD_NUM_WARPS = 4
_BWD_NUM_WARPS = 4
_BLOCK_J = 128
_BWD_BLOCK = 128

DEFAULT_RADIUS_PROTEIN = 15.0
DEFAULT_RADIUS_NUCLEOTIDE = 30.0
DEFAULT_NEIGHBOR_CAP_PROTEIN = 512
DEFAULT_NEIGHBOR_CAP_NUCLEOTIDE = 2048


def is_ball_query_triton_installed() -> bool:
    """Return whether Triton is importable."""
    return _TRITON_AVAILABLE


def is_ball_query_triton_available() -> bool:
    """Return whether this backend can run (Triton + CUDA)."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _as_soa(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[N, P, 3]`` into contiguous x/y/z planes."""
    return xyz[..., 0].contiguous(), xyz[..., 1].contiguous(), xyz[..., 2].contiguous()


if _TRITON_AVAILABLE:

    @triton.jit
    def _hash_rng(state):
        # Advance before mixing so zero is not an absorbing state.
        state = state + 0x9E3779B9
        state = state ^ (state >> 16)
        state = state * 0x45D9F3B
        state = state ^ (state >> 16)
        state = state * 0x45D9F3B
        state = state ^ (state >> 16)
        return state

    @triton.jit(
        do_not_specialize=[
            "N",
            "P1",
            "P2",
            "K",
            "radius_protein",
            "radius_nucleotide",
            "seed",
        ]
    )
    def _ball_query_forward_kernel(
        p1x_ptr,
        p1y_ptr,
        p1z_ptr,
        p2x_ptr,
        p2y_ptr,
        p2z_ptr,
        predx_ptr,
        predy_ptr,
        predz_ptr,
        is_nuc_ptr,
        query_idx_ptr,
        lengths1_ptr,
        lengths2_ptr,
        idx_ptr,
        dists_gt_ptr,
        dists_pred_ptr,
        neighbor_count_ptr,
        N,
        P1,
        P2,
        K,
        radius_protein,
        radius_nucleotide,
        seed,
        PRED_IS_BF16: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """One program / query atom: SoA tiled scan + Algorithm R."""
        atom_flat = tl.program_id(0).to(tl.int64)
        total_atoms = N * P1
        if atom_flat >= total_atoms:
            return

        n = atom_flat // P1
        i = atom_flat % P1

        len1 = tl.load(lengths1_ptr + n)
        if i >= len1:
            return

        len2 = tl.load(lengths2_ptr + n)

        qi = tl.load(query_idx_ptr + n * P1 + i)
        is_nuc = tl.load(is_nuc_ptr + n * P1 + i)
        radius = tl.where(is_nuc > 0.5, radius_nucleotide, radius_protein)
        radius2 = radius * radius

        qi_base = n * P1 + i
        pred_qi_base = n * P2 + qi
        qi_x = tl.load(p1x_ptr + qi_base)
        qi_y = tl.load(p1y_ptr + qi_base)
        qi_z = tl.load(p1z_ptr + qi_base)

        if PRED_IS_BF16:
            pi_x = tl.load(predx_ptr + pred_qi_base).to(tl.float32)
            pi_y = tl.load(predy_ptr + pred_qi_base).to(tl.float32)
            pi_z = tl.load(predz_ptr + pred_qi_base).to(tl.float32)
        else:
            pi_x = tl.load(predx_ptr + pred_qi_base)
            pi_y = tl.load(predy_ptr + pred_qi_base)
            pi_z = tl.load(predz_ptr + pred_qi_base)

        out_base = (n * P1 + i) * K
        row_base = n * P2

        count = tl.full((), 0, tl.int32)
        seen = tl.full((), 0, tl.int32)
        rng = seed.to(tl.uint32)
        rng = rng ^ (n.to(tl.uint32) * tl.full((), 1000003, tl.uint32))
        rng = rng ^ (i.to(tl.uint32) * tl.full((), 997, tl.uint32))
        rng = _hash_rng(rng)

        for j0 in tl.range(0, len2, BLOCK_J):
            offs = j0 + tl.arange(0, BLOCK_J)
            in_range = offs < len2
            j_ptr = row_base + offs

            xj = tl.load(p2x_ptr + j_ptr, mask=in_range, other=1.0e6)
            yj = tl.load(p2y_ptr + j_ptr, mask=in_range, other=1.0e6)
            zj = tl.load(p2z_ptr + j_ptr, mask=in_range, other=1.0e6)

            dx = qi_x - xj
            dy = qi_y - yj
            dz = qi_z - zj
            dist2_gt = dx * dx + dy * dy + dz * dz
            in_ball = (
                in_range
                & (offs != qi)
                & (tl.abs(dx) <= radius)
                & (tl.abs(dy) <= radius)
                & (tl.abs(dz) <= radius)
                & (dist2_gt < radius2)
            )

            hits_i32 = in_ball.to(tl.int32)
            n_hits = tl.sum(hits_i32)

            if n_hits > 0 and count < K:
                rem = K - count
                excl = tl.cumsum(hits_i32) - hits_i32
                take = in_ball & (excl < rem)
                write_slot = count + excl

                if PRED_IS_BF16:
                    px = tl.load(predx_ptr + j_ptr, mask=take, other=0.0).to(tl.float32)
                    py = tl.load(predy_ptr + j_ptr, mask=take, other=0.0).to(tl.float32)
                    pz = tl.load(predz_ptr + j_ptr, mask=take, other=0.0).to(tl.float32)
                else:
                    px = tl.load(predx_ptr + j_ptr, mask=take, other=0.0)
                    py = tl.load(predy_ptr + j_ptr, mask=take, other=0.0)
                    pz = tl.load(predz_ptr + j_ptr, mask=take, other=0.0)
                v_pdx = pi_x - px
                v_pdy = pi_y - py
                v_pdz = pi_z - pz
                v_dist2_pred = v_pdx * v_pdx + v_pdy * v_pdy + v_pdz * v_pdz

                tl.store(idx_ptr + out_base + write_slot, offs, mask=take)
                tl.store(dists_gt_ptr + out_base + write_slot, dist2_gt, mask=take)
                if PRED_IS_BF16:
                    tl.store(
                        dists_pred_ptr + out_base + write_slot,
                        v_dist2_pred.to(tl.bfloat16),
                        mask=take,
                    )
                else:
                    tl.store(
                        dists_pred_ptr + out_base + write_slot,
                        v_dist2_pred,
                        mask=take,
                    )

                n_take = tl.minimum(n_hits, rem)
                count = count + n_take

                if n_hits > n_take:
                    hit_ord = 0
                    for b in tl.static_range(BLOCK_J):
                        j = j0 + b
                        ok = j < len2
                        jj = row_base + j
                        x = tl.load(p2x_ptr + jj, mask=ok, other=1.0e6)
                        y = tl.load(p2y_ptr + jj, mask=ok, other=1.0e6)
                        z = tl.load(p2z_ptr + jj, mask=ok, other=1.0e6)
                        ddx = qi_x - x
                        ddy = qi_y - y
                        ddz = qi_z - z
                        d2g = ddx * ddx + ddy * ddy + ddz * ddz
                        ib = (
                            ok
                            & (j != qi)
                            & (tl.abs(ddx) <= radius)
                            & (tl.abs(ddy) <= radius)
                            & (tl.abs(ddz) <= radius)
                            & (d2g < radius2)
                        )
                        if ib:
                            hit_ord = hit_ord + 1
                            seen = seen + 1
                            if hit_ord > n_take:
                                if PRED_IS_BF16:
                                    s_px = tl.load(predx_ptr + jj).to(tl.float32)
                                    s_py = tl.load(predy_ptr + jj).to(tl.float32)
                                    s_pz = tl.load(predz_ptr + jj).to(tl.float32)
                                else:
                                    s_px = tl.load(predx_ptr + jj)
                                    s_py = tl.load(predy_ptr + jj)
                                    s_pz = tl.load(predz_ptr + jj)
                                s_pdx = pi_x - s_px
                                s_pdy = pi_y - s_py
                                s_pdz = pi_z - s_pz
                                s_d2p = s_pdx * s_pdx + s_pdy * s_pdy + s_pdz * s_pdz
                                rng = _hash_rng(rng)
                                r = (rng % seen.to(tl.uint32)).to(tl.int32)
                                if r < K:
                                    slot = r.to(tl.int64)
                                    tl.store(idx_ptr + out_base + slot, j)
                                    tl.store(dists_gt_ptr + out_base + slot, d2g)
                                    if PRED_IS_BF16:
                                        tl.store(
                                            dists_pred_ptr + out_base + slot,
                                            s_d2p.to(tl.bfloat16),
                                        )
                                    else:
                                        tl.store(
                                            dists_pred_ptr + out_base + slot, s_d2p
                                        )
                else:
                    seen = seen + n_hits

            elif n_hits > 0:
                # Reservoir phase (count == K).
                for b in tl.static_range(BLOCK_J):
                    j = j0 + b
                    ok = j < len2
                    jj = row_base + j
                    x = tl.load(p2x_ptr + jj, mask=ok, other=1.0e6)
                    y = tl.load(p2y_ptr + jj, mask=ok, other=1.0e6)
                    z = tl.load(p2z_ptr + jj, mask=ok, other=1.0e6)
                    ddx = qi_x - x
                    ddy = qi_y - y
                    ddz = qi_z - z
                    d2g = ddx * ddx + ddy * ddy + ddz * ddz
                    ib = (
                        ok
                        & (j != qi)
                        & (tl.abs(ddx) <= radius)
                        & (tl.abs(ddy) <= radius)
                        & (tl.abs(ddz) <= radius)
                        & (d2g < radius2)
                    )
                    if ib:
                        seen = seen + 1
                        if PRED_IS_BF16:
                            s_px = tl.load(predx_ptr + jj).to(tl.float32)
                            s_py = tl.load(predy_ptr + jj).to(tl.float32)
                            s_pz = tl.load(predz_ptr + jj).to(tl.float32)
                        else:
                            s_px = tl.load(predx_ptr + jj)
                            s_py = tl.load(predy_ptr + jj)
                            s_pz = tl.load(predz_ptr + jj)
                        s_pdx = pi_x - s_px
                        s_pdy = pi_y - s_py
                        s_pdz = pi_z - s_pz
                        s_d2p = s_pdx * s_pdx + s_pdy * s_pdy + s_pdz * s_pdz
                        rng = _hash_rng(rng)
                        r = (rng % seen.to(tl.uint32)).to(tl.int32)
                        if r < K:
                            slot = r.to(tl.int64)
                            tl.store(idx_ptr + out_base + slot, j)
                            tl.store(dists_gt_ptr + out_base + slot, d2g)
                            if PRED_IS_BF16:
                                tl.store(
                                    dists_pred_ptr + out_base + slot,
                                    s_d2p.to(tl.bfloat16),
                                )
                            else:
                                tl.store(dists_pred_ptr + out_base + slot, s_d2p)

        tl.store(neighbor_count_ptr + n * P1 + i, seen)

    @triton.jit(do_not_specialize=["N", "P1", "P2", "K", "total"])
    def _ball_query_backward_kernel(
        predx_ptr,
        predy_ptr,
        predz_ptr,
        idx_ptr,
        query_idx_ptr,
        grad_dists_ptr,
        grad_pred_ptr,
        N,
        P1,
        P2,
        K,
        total,
        PRED_IS_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        start = pid * BLOCK
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total

        pk = P1 * K
        n = offs // pk
        rem = offs % pk
        i = rem // K

        j = tl.load(idx_ptr + offs, mask=mask, other=-1)
        qi = tl.load(query_idx_ptr + n * P1 + i, mask=mask, other=-1)
        g = tl.load(grad_dists_ptr + offs, mask=mask, other=0.0)
        active = mask & (j >= 0) & (qi >= 0) & (g != 0.0)

        base_i = n * P2 + qi
        base_j = n * P2 + j
        # grad_pred is still AoS [N,P,3]
        gbase_i = base_i * 3
        gbase_j = base_j * 3

        if PRED_IS_BF16:
            pi_x = tl.load(predx_ptr + base_i, mask=active, other=0.0).to(tl.float32)
            pi_y = tl.load(predy_ptr + base_i, mask=active, other=0.0).to(tl.float32)
            pi_z = tl.load(predz_ptr + base_i, mask=active, other=0.0).to(tl.float32)
            pj_x = tl.load(predx_ptr + base_j, mask=active, other=0.0).to(tl.float32)
            pj_y = tl.load(predy_ptr + base_j, mask=active, other=0.0).to(tl.float32)
            pj_z = tl.load(predz_ptr + base_j, mask=active, other=0.0).to(tl.float32)
        else:
            pi_x = tl.load(predx_ptr + base_i, mask=active, other=0.0)
            pi_y = tl.load(predy_ptr + base_i, mask=active, other=0.0)
            pi_z = tl.load(predz_ptr + base_i, mask=active, other=0.0)
            pj_x = tl.load(predx_ptr + base_j, mask=active, other=0.0)
            pj_y = tl.load(predy_ptr + base_j, mask=active, other=0.0)
            pj_z = tl.load(predz_ptr + base_j, mask=active, other=0.0)

        gx = 2.0 * g * (pi_x - pj_x)
        gy = 2.0 * g * (pi_y - pj_y)
        gz = 2.0 * g * (pi_z - pj_z)
        tl.atomic_add(grad_pred_ptr + gbase_i + 0, gx, mask=active)
        tl.atomic_add(grad_pred_ptr + gbase_i + 1, gy, mask=active)
        tl.atomic_add(grad_pred_ptr + gbase_i + 2, gz, mask=active)
        tl.atomic_add(grad_pred_ptr + gbase_j + 0, -gx, mask=active)
        tl.atomic_add(grad_pred_ptr + gbase_j + 1, -gy, mask=active)
        tl.atomic_add(grad_pred_ptr + gbase_j + 2, -gz, mask=active)


def _query_with_pred_compact(
    p1: torch.Tensor,
    p2: torch.Tensor,
    pred: torch.Tensor,
    is_nucleotide: torch.Tensor,
    query_indices: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    k: int,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query compact rows; return indices, squared distances, and true counts."""
    if not is_ball_query_triton_available():
        raise RuntimeError(
            "Triton ball-query smooth lDDT requires Triton and a CUDA device"
        )
    if not (p1.is_cuda and p2.is_cuda and pred.is_cuda):
        raise RuntimeError("Triton ball-query requires CUDA tensors")
    if p1.dtype != torch.float32 or p2.dtype != torch.float32:
        raise RuntimeError("p1/p2 must be float32")
    if pred.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("pred must be float32 or bfloat16")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    p1 = p1.contiguous()
    p2 = p2.contiguous()
    pred = pred.contiguous()
    is_nucleotide = is_nucleotide.contiguous().to(dtype=torch.float32)
    query_indices = query_indices.contiguous().long()
    lengths1 = lengths1.contiguous().long()
    lengths2 = lengths2.contiguous().long()

    n_batch, n_query, _ = p1.shape
    n_atom = p2.shape[1]
    idxs = torch.full((n_batch, n_query, k), -1, device=p1.device, dtype=torch.long)
    dists_gt = torch.zeros((n_batch, n_query, k), device=p1.device, dtype=torch.float32)
    dists_pred = torch.zeros(
        (n_batch, n_query, k), device=pred.device, dtype=pred.dtype
    )
    neighbor_count = torch.zeros(
        (n_batch, n_query), device=p1.device, dtype=torch.int32
    )

    total_atoms = n_batch * n_query
    if total_atoms == 0 or idxs.numel() == 0:
        return idxs, dists_gt, dists_pred, neighbor_count

    p1x, p1y, p1z = _as_soa(p1)
    p2x, p2y, p2z = _as_soa(p2)
    predx, predy, predz = _as_soa(pred)

    _ball_query_forward_kernel[(total_atoms,)](
        p1x,
        p1y,
        p1z,
        p2x,
        p2y,
        p2z,
        predx,
        predy,
        predz,
        is_nucleotide,
        query_indices,
        lengths1,
        lengths2,
        idxs,
        dists_gt,
        dists_pred,
        neighbor_count,
        n_batch,
        n_query,
        n_atom,
        k,
        float(radius_protein),
        float(radius_nucleotide),
        int(seed) & 0xFFFFFFFF,
        PRED_IS_BF16=pred.dtype == torch.bfloat16,
        BLOCK_J=_BLOCK_J,
        num_warps=_FWD_NUM_WARPS,
    )
    return idxs, dists_gt, dists_pred, neighbor_count


def _query_with_pred(
    p1: torch.Tensor,
    p2: torch.Tensor,
    pred: torch.Tensor,
    is_nucleotide: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    k: int,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible full-row query entry point."""
    n_batch, n_query, _ = p1.shape
    query_indices = torch.arange(n_query, device=p1.device).expand(n_batch, -1)
    idx, dists_gt, dists_pred, _ = _query_with_pred_compact(
        p1=p1,
        p2=p2,
        pred=pred,
        is_nucleotide=is_nucleotide,
        query_indices=query_indices,
        lengths1=lengths1,
        lengths2=lengths2,
        k=k,
        radius_protein=radius_protein,
        radius_nucleotide=radius_nucleotide,
        seed=seed,
    )
    return idx, dists_gt, dists_pred


def _pred_backward(
    pred: torch.Tensor,
    idxs: torch.Tensor,
    query_indices: torch.Tensor,
    grad_dists: torch.Tensor,
) -> torch.Tensor:
    """Backward for predicted squared distances (fp32 atomic scatter)."""
    if not is_ball_query_triton_available():
        raise RuntimeError(
            "Triton ball-query smooth lDDT requires Triton and a CUDA device"
        )
    if not pred.is_cuda:
        raise RuntimeError("Triton ball-query requires CUDA tensors")

    pred = pred.contiguous()
    idxs = idxs.contiguous().long()
    query_indices = query_indices.contiguous().long()
    grad_dists = grad_dists.contiguous().float()

    n_batch, n_atom, _ = pred.shape
    n_query = idxs.shape[1]
    k = idxs.shape[-1]
    grad_pred = torch.zeros(
        (n_batch, n_atom, 3), device=pred.device, dtype=torch.float32
    )

    total = n_batch * n_query * k
    if total == 0:
        return grad_pred

    predx, predy, predz = _as_soa(pred)
    n_programs = (total + _BWD_BLOCK - 1) // _BWD_BLOCK
    _ball_query_backward_kernel[(n_programs,)](
        predx,
        predy,
        predz,
        idxs,
        query_indices,
        grad_dists,
        grad_pred,
        n_batch,
        n_query,
        n_atom,
        k,
        total,
        PRED_IS_BF16=pred.dtype == torch.bfloat16,
        BLOCK=_BWD_BLOCK,
        num_warps=_BWD_NUM_WARPS,
    )
    return grad_pred


def _expand_to_x_batch(
    tensor: torch.Tensor, x: torch.Tensor, trailing_dims: int
) -> torch.Tensor:
    while tensor.ndim < len(x.shape[:-2]) + trailing_dims:
        tensor = tensor.unsqueeze(-trailing_dims - 1)
    return tensor.expand(*x.shape[:-2], *tensor.shape[-trailing_dims:])


def _flatten_batch(tensor: torch.Tensor, trailing_dims: int) -> torch.Tensor:
    return tensor.reshape(-1, *tensor.shape[-trailing_dims:])


def _validate_top_k(top_k: int | None, n_atom: int) -> int:
    if top_k is None:
        raise ValueError("smooth_lddt_top_k must be set for ball-query smooth lDDT")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("smooth_lddt_top_k must be a positive integer")
    return min(top_k, max(n_atom - 1, 0))


def _validate_nucleotide_scale(nucleotide_scale: float) -> float:
    if (
        isinstance(nucleotide_scale, bool)
        or not isinstance(nucleotide_scale, int | float)
        or not math.isfinite(nucleotide_scale)
        or nucleotide_scale < 1.0
    ):
        raise ValueError("smooth_lddt_nucleotide_scale must be finite and >= 1")
    return float(nucleotide_scale)


def _type_aware_top_ks(top_k: int, nucleotide_scale: float) -> tuple[int, int]:
    """Return non-nucleotide and nucleotide capacities for a reference ``K``."""
    protein_top_k = max(1, math.ceil(top_k / nucleotide_scale))
    return protein_top_k, top_k


class _BallQueryWithPredDist(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pred,
        p1_gt,
        p2_gt,
        is_nucleotide,
        query_indices,
        lengths1,
        lengths2,
        k,
        radius_protein,
        radius_nucleotide,
        seed,
    ):
        idx, dists_gt, dists_pred, neighbor_count = _query_with_pred_compact(
            p1=p1_gt,
            p2=p2_gt,
            pred=pred,
            is_nucleotide=is_nucleotide,
            query_indices=query_indices,
            lengths1=lengths1,
            lengths2=lengths2,
            k=k,
            radius_protein=radius_protein,
            radius_nucleotide=radius_nucleotide,
            seed=seed,
        )
        ctx.save_for_backward(pred, idx, query_indices)
        ctx.mark_non_differentiable(idx, dists_gt, neighbor_count)
        return dists_pred, dists_gt, idx, neighbor_count

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_dists_pred, grad_dists_gt, grad_idx, grad_neighbor_count):
        pred, idx, query_indices = ctx.saved_tensors
        grad_pred = _pred_backward(pred, idx, query_indices, grad_dists_pred).to(
            pred.dtype
        )
        return (
            grad_pred,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _score_from_dists(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    valid_neighbor: torch.Tensor,
    loss_atom_mask_i: torch.Tensor,
    loss_atom_mask_j: torch.Tensor,
    flat_is_nucleotide: torch.Tensor,
    eps: float,
    out_shape: tuple[int, ...],
    radius_protein: float,
    radius_nucleotide: float,
) -> torch.Tensor:
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
        torch.tensor(radius_nucleotide, device=dx.device, dtype=dx_gt.dtype),
        torch.tensor(radius_protein, device=dx.device, dtype=dx_gt.dtype),
    )
    c = dx_gt < cutoff[..., None]
    mask = (valid_neighbor * loss_atom_mask_i * loss_atom_mask_j).bool()
    mask_sum = torch.sum(mask, dim=(-1, -2)) + eps
    ce_mean = torch.sum(c * e * mask, dim=(-1, -2)) / mask_sum
    c_mean = torch.sum(c * mask, dim=(-1, -2)) / mask_sum
    lddt = ce_mean / (c_mean + eps)
    return (1 - lddt).reshape(out_shape)


def _legacy_ball_query_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    top_k: int | None = 512,
    seed: int = 0,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
) -> torch.Tensor:
    """Legacy ball-query smooth lDDT retained for benchmark comparisons.

    Same argument contract as the CUDA package entry point. Radii default to
    15 Å (protein) / 30 Å (nucleotide).
    """
    if not is_ball_query_triton_available():
        raise RuntimeError(
            "Triton ball-query smooth lDDT requires Triton and a CUDA device"
        )
    if not x.is_cuda:
        raise RuntimeError("Ball-query smooth lDDT requires CUDA tensors")
    if radius_protein <= 0 or radius_nucleotide <= 0:
        raise ValueError("radius_protein and radius_nucleotide must be positive")

    n_atom = x.shape[-2]
    k = _validate_top_k(top_k=top_k, n_atom=n_atom)
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

    p2 = torch.where(
        flat_atom_mask_gt[..., None].bool(),
        flat_x_gt,
        torch.full_like(flat_x_gt, 1.0e6),
    )

    query_indices = torch.arange(n_atom, device=x.device).expand(flat_batch, -1)
    dists_pred, dists_gt, idx, _ = _BallQueryWithPredDist.apply(
        flat_x,
        flat_x_gt,
        p2,
        flat_is_nucleotide,
        query_indices,
        lengths,
        lengths,
        k,
        float(radius_protein),
        float(radius_nucleotide),
        seed,
    )

    safe_idx = idx.clamp_min(0)
    flat_safe_idx = safe_idx.reshape(flat_batch, -1)
    valid_neighbor = idx >= 0
    loss_atom_mask_i = flat_loss_atom_mask[..., None]
    loss_atom_mask_j = torch.gather(
        flat_loss_atom_mask, dim=1, index=flat_safe_idx
    ).reshape(flat_batch, n_atom, k)

    out_shape = x.shape[:-2]
    score_args = (
        dists_pred,
        dists_gt,
        valid_neighbor,
        loss_atom_mask_i,
        loss_atom_mask_j,
        flat_is_nucleotide,
        eps,
        out_shape,
        float(radius_protein),
        float(radius_nucleotide),
    )
    if dists_pred.requires_grad:
        return torch.utils.checkpoint.checkpoint(
            _score_from_dists,
            *score_args,
            use_reentrant=False,
        )
    return _score_from_dists(*score_args)


def _compact_query_group(
    coordinates: torch.Tensor,
    query_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact selected query atoms and retain their global atom indices."""
    lengths = query_mask.sum(dim=1, dtype=torch.long)
    max_queries = int(lengths.max().item()) if lengths.numel() else 0
    if max_queries == 0:
        shape = (coordinates.shape[0], 0)
        return (
            coordinates[:, :0],
            torch.empty(shape, device=coordinates.device, dtype=torch.long),
            is_nucleotide[:, :0],
            lengths,
        )

    order = torch.argsort((~query_mask).to(torch.int8), dim=1, stable=True)
    query_indices = order[:, :max_queries]
    gather_idx = query_indices[..., None].expand(-1, -1, 3)
    query_coordinates = torch.gather(coordinates, 1, gather_idx)
    query_is_nucleotide = torch.gather(is_nucleotide, 1, query_indices)
    return query_coordinates, query_indices, query_is_nucleotide, lengths


def _query_group_with_pred(
    flat_x: torch.Tensor,
    flat_x_gt: torch.Tensor,
    valid_atom: torch.Tensor,
    flat_is_nucleotide: torch.Tensor,
    query_mask: torch.Tensor,
    top_k: int,
    query_radius_protein: float,
    query_radius_nucleotide: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    query_gt, query_indices, query_is_nucleotide, query_lengths = _compact_query_group(
        flat_x_gt, query_mask, flat_is_nucleotide
    )
    if query_gt.shape[1] == 0:
        empty = flat_x.new_empty((flat_x.shape[0], 0, top_k))
        return (
            empty,
            empty.float(),
            torch.empty_like(empty, dtype=torch.long),
            torch.empty(empty.shape[:-1], device=flat_x.device, dtype=torch.int32),
            query_indices,
            query_is_nucleotide,
        )

    candidates_gt = torch.where(
        valid_atom[..., None], flat_x_gt, torch.full_like(flat_x_gt, 1.0e6)
    )
    candidate_lengths = torch.full(
        (flat_x.shape[0],),
        flat_x.shape[1],
        device=flat_x.device,
        dtype=torch.long,
    )
    dists_pred, dists_gt, idx, neighbor_count = _BallQueryWithPredDist.apply(
        flat_x,
        query_gt,
        candidates_gt,
        query_is_nucleotide,
        query_indices,
        query_lengths,
        candidate_lengths,
        top_k,
        float(query_radius_protein),
        float(query_radius_nucleotide),
        seed,
    )
    return (
        dists_pred,
        dists_gt,
        idx,
        neighbor_count,
        query_indices,
        query_is_nucleotide,
    )


def _top_k_adjusted_type_group_weights(
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return dynamically scaled type weights with a lower bound of one."""
    protein_weight = max(
        1.0,
        DEFAULT_NEIGHBOR_CAP_PROTEIN / top_k,
    )
    nucleotide_weight = max(
        1.0,
        DEFAULT_NEIGHBOR_CAP_NUCLEOTIDE / top_k,
    )
    return torch.tensor(
        (1.0, protein_weight, nucleotide_weight),
        device=device,
        dtype=dtype,
    )


def _top_k_adjusted_type_row_weights(
    neighbor_count: torch.Tensor,
    retained_count: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    top_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return dynamically scaled, lower-bounded weights for truncated rows."""
    group_weight = _top_k_adjusted_type_group_weights(
        top_k, neighbor_count.device, dtype
    )
    adjusted_weight = torch.where(
        query_is_nucleotide.bool(), group_weight[2], group_weight[1]
    )
    truncated = neighbor_count > retained_count
    return torch.where(truncated, adjusted_weight, group_weight[0])


def _type_group_indices(
    neighbor_count: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Map rows to exact=0, truncated protein=1, or truncated nucleotide=2."""
    truncated = (neighbor_count > top_k).long()
    return truncated * (1 + query_is_nucleotide.long())


def _group_row_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce retained pairs to one score and count per query atom."""
    dx = torch.sqrt(eps + dists_pred.clamp_min(0.0))
    dx_gt = torch.sqrt(eps + dists_gt.clamp_min(0.0))
    delta = torch.abs(dx_gt - dx)
    score = 0.25 * (
        torch.sigmoid(0.5 - delta)
        + torch.sigmoid(1.0 - delta)
        + torch.sigmoid(2.0 - delta)
        + torch.sigmoid(4.0 - delta)
    )
    cutoff = torch.where(
        query_is_nucleotide.bool(),
        torch.as_tensor(score_radius_nucleotide, device=dx.device, dtype=dx_gt.dtype),
        torch.as_tensor(score_radius_protein, device=dx.device, dtype=dx_gt.dtype),
    )
    retained = idx >= 0
    valid = retained & (dx_gt < cutoff[..., None])
    return (
        torch.sum(score * valid.to(score.dtype), dim=-1),
        torch.sum(valid, dim=-1, dtype=score.dtype),
    )


def _type_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    type_group_index: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate query rows into the three reweighting categories."""
    row_score_sum, row_pair_count = _group_row_sums(
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    out_shape = (*row_score_sum.shape[:-1], 3)
    score_sum = torch.zeros(
        out_shape, device=row_score_sum.device, dtype=row_score_sum.dtype
    )
    pair_count = torch.zeros_like(score_sum)
    return (
        score_sum.scatter_add(-1, type_group_index, row_score_sum),
        pair_count.scatter_add(-1, type_group_index, row_pair_count),
    )


def _combine_type_group_sums(
    score_sum: torch.Tensor,
    pair_count: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the three scalar category weights after pair reduction."""
    group_weight = _top_k_adjusted_type_group_weights(
        top_k, score_sum.device, score_sum.dtype
    )
    return (
        torch.sum(score_sum * group_weight, dim=-1),
        torch.sum(pair_count * group_weight, dim=-1),
    )


def _top_k_adjusted_type_reweighted_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    neighbor_count: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    top_k: int,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply type weights to three aggregate statistics."""
    type_group_index = _type_group_indices(neighbor_count, query_is_nucleotide, top_k)
    score_sum, pair_count = _type_group_sums(
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        type_group_index,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    return _combine_type_group_sums(score_sum, pair_count, top_k)


def _unweighted_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_score_sum, row_pair_count = _group_row_sums(
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    return row_score_sum.sum(dim=-1), row_pair_count.sum(dim=-1)


def _checkpointed_group_row_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    args = (
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    if dists_pred.requires_grad:
        return torch.utils.checkpoint.checkpoint(
            _group_row_sums, *args, use_reentrant=False
        )
    return _group_row_sums(*args)


def _checkpointed_type_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    type_group_index: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    args = (
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        type_group_index,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    if dists_pred.requires_grad:
        return torch.utils.checkpoint.checkpoint(
            _type_group_sums, *args, use_reentrant=False
        )
    return _type_group_sums(*args)


def _checkpointed_top_k_adjusted_type_reweighted_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    neighbor_count: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    top_k: int,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    type_group_index = _type_group_indices(neighbor_count, query_is_nucleotide, top_k)
    score_sum, pair_count = _checkpointed_type_group_sums(
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        type_group_index,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    return _combine_type_group_sums(score_sum, pair_count, top_k)


def _checkpointed_unweighted_group_sums(
    dists_pred: torch.Tensor,
    dists_gt: torch.Tensor,
    idx: torch.Tensor,
    query_is_nucleotide: torch.Tensor,
    eps: float,
    score_radius_protein: float,
    score_radius_nucleotide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_score_sum, row_pair_count = _checkpointed_group_row_sums(
        dists_pred,
        dists_gt,
        idx,
        query_is_nucleotide,
        eps,
        score_radius_protein,
        score_radius_nucleotide,
    )
    return row_score_sum.sum(dim=-1), row_pair_count.sum(dim=-1)


def _finish_weighted_lddt(
    score_sum: torch.Tensor,
    pair_count: torch.Tensor,
    valid_atom: torch.Tensor,
    eps: float,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    valid_count = valid_atom.sum(dim=-1).to(score_sum.dtype)
    all_pair_count = valid_count * (valid_count - 1).clamp_min(0)
    normalizer = all_pair_count + eps
    score_mean = score_sum / normalizer
    local_pair_mean = pair_count / normalizer
    return (1 - score_mean / (local_pair_mean + eps)).reshape(out_shape)


def _prepare_sparse_inputs(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    x_gt = _expand_to_x_batch(x_gt, x, trailing_dims=2)
    atom_mask_gt = _expand_to_x_batch(atom_mask_gt, x, trailing_dims=1)
    is_nucleotide = _expand_to_x_batch(is_nucleotide, x, trailing_dims=1)
    loss_atom_mask = _expand_to_x_batch(loss_atom_mask, x, trailing_dims=1)
    valid_atom = (loss_atom_mask * atom_mask_gt).bool()
    return (
        _flatten_batch(x, trailing_dims=2),
        _flatten_batch(x_gt, trailing_dims=2),
        _flatten_batch(valid_atom, trailing_dims=1),
        _flatten_batch(is_nucleotide, trailing_dims=1),
    )


def ball_query_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    top_k: int | None = 512,
    seed: int = 0,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
) -> torch.Tensor:
    """Estimate smooth lDDT with top-k-adjusted type reweighting."""
    if not is_ball_query_triton_available() or not x.is_cuda:
        raise RuntimeError(
            "Triton ball-query smooth lDDT requires Triton and a CUDA device"
        )
    if radius_protein <= 0 or radius_nucleotide <= 0:
        raise ValueError("radius_protein and radius_nucleotide must be positive")
    k = _validate_top_k(top_k, x.shape[-2])
    if k == 0:
        return torch.ones(x.shape[:-2], device=x.device, dtype=x.dtype) + x.sum() * 0
    flat_x, flat_x_gt, valid_atom, flat_is_nucleotide = _prepare_sparse_inputs(
        x, x_gt, atom_mask_gt, is_nucleotide, loss_atom_mask
    )
    group = _query_group_with_pred(
        flat_x,
        flat_x_gt,
        valid_atom,
        flat_is_nucleotide,
        valid_atom,
        k,
        radius_protein,
        radius_nucleotide,
        seed,
    )
    score_sum, pair_count = _checkpointed_top_k_adjusted_type_reweighted_group_sums(
        group[0],
        group[1],
        group[2],
        group[3],
        group[5],
        k,
        eps,
        radius_protein,
        radius_nucleotide,
    )
    return _finish_weighted_lddt(score_sum, pair_count, valid_atom, eps, x.shape[:-2])


def _type_aware_ball_query_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    top_k: int | None = 2048,
    nucleotide_scale: float = 4.0,
    seed: int = 0,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
) -> torch.Tensor:
    """Use ``K / s`` non-nucleotide and ``K`` nucleotide neighbors."""
    if not is_ball_query_triton_available() or not x.is_cuda:
        raise RuntimeError(
            "Triton ball-query smooth lDDT requires Triton and a CUDA device"
        )
    if radius_protein <= 0 or radius_nucleotide <= 0:
        raise ValueError("radius_protein and radius_nucleotide must be positive")
    scale = _validate_nucleotide_scale(nucleotide_scale)
    nucleotide_top_k = _validate_top_k(top_k, x.shape[-2])
    if nucleotide_top_k == 0:
        return torch.ones(x.shape[:-2], device=x.device, dtype=x.dtype) + x.sum() * 0
    protein_top_k, nucleotide_top_k = _type_aware_top_ks(nucleotide_top_k, scale)
    flat_x, flat_x_gt, valid_atom, flat_is_nucleotide = _prepare_sparse_inputs(
        x, x_gt, atom_mask_gt, is_nucleotide, loss_atom_mask
    )

    score_sum = torch.zeros(flat_x.shape[0], device=x.device, dtype=x.dtype)
    pair_count = torch.zeros_like(score_sum)
    query_groups = (
        (valid_atom & ~flat_is_nucleotide.bool(), protein_top_k),
        (valid_atom & flat_is_nucleotide.bool(), nucleotide_top_k),
    )
    for group_index, (query_mask, group_top_k) in enumerate(query_groups):
        group = _query_group_with_pred(
            flat_x,
            flat_x_gt,
            valid_atom,
            flat_is_nucleotide,
            query_mask,
            group_top_k,
            radius_protein,
            radius_nucleotide,
            seed + group_index,
        )
        group_score_sum, group_pair_count = _checkpointed_unweighted_group_sums(
            group[0],
            group[1],
            group[2],
            group[5],
            eps,
            radius_protein,
            radius_nucleotide,
        )
        score_sum = score_sum + group_score_sum
        pair_count = pair_count + group_pair_count
    return _finish_weighted_lddt(score_sum, pair_count, valid_atom, eps, x.shape[:-2])
