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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: low-memory streaming template-pair
# embedding with chunked normalization and in-place projection accumulation.

"""Fused-primitive dispatcher for the template pair embedder.

Design rationale:

We keep cuBLAS for every matmul (fp32 IEEE is bandwidth-bound but cuBLAS
still beats any Triton alternative we've tuned for these shapes) and only
restructure the surrounding ops to eliminate persistent transients:

1. **Chunked LN → matmul into ``a``**. Rather than materialize the full
   ``z_projected = linear_z(layer_norm_z(z))`` at 0.5U, tile ``z`` into
   64-row chunks: LN one chunk (0.05U scratch), then write
   ``F.linear(chunk_norm, W_z, B_z)`` directly into ``a[:, start:end]``.
   Persistent transient: 0.05U (LN chunk), amortized to 0 as it rises
   and falls each iteration.
2. **In-place addmm** for the dgram projection. ``torch.addmm(out=a_flat)``
   accumulates into ``a`` without allocating a separate 0.5U output.
3. **Small [B, N, 64] restype projections + broadcast-add**. Rather than
   materialize ``restype_i.expand(-1, N, -1).to(dtype)`` at 0.25U each,
   compute the tiny 2D projection once (``[B, N, 32] @ [32, 64] -> [B,
   N, 64]``, ~0.001U) and broadcast-add into ``a``.
4. **Materialize scalar_feats once** (5 channels, ~0.04U) and in-place
   addmm into ``a``.

Net per-call transient above ``a``: the ``dgram`` slice (~0.15U at
c_dgram=39) plus the ~0.04U scalar cat plus the 0.05U LN chunk. Eager
path allocates 0.5U (dgram_linear out) + 2×0.25U (aatype .to expand) +
0.5U (z_projected) ≈ **1.5U transient above a**. Fused path:
**~0.25U above a** at typical N.

Eligibility (all must hold):
* ``OPENFOLD3_FUSED_TEMPLATE_EMBED != "0"`` (default-on when Triton is available)
* CUDA + Triton available
* Inference-only (no gradient tracking)

On any failure the caller runs the eager path unchanged.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold3.core.kernels.triton.fused_ln_linear import is_triton_available
from openfold3.core.kernels.triton.fused_template_coordinate import (
    template_coordinate_projection_add_,
)


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val in {"1", "true", "yes", "on"}


def is_fused_template_pair_embedder_enabled() -> bool:
    """Runtime gate. Default-on when Triton is available."""
    if not is_triton_available():
        return False
    return _env_flag("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")


def _build_scalar_feats(
    batch: dict,
    template_index: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize ``[B, N, N, 5]`` scalar feats for template ``template_index``.

    Mirrors the ``torch.cat`` block inside
    ``TemplatePairEmbedderAllAtom._embed_feats`` for a single template so the
    tensor stays tiny (~0.04U at typical N).
    """
    ti = template_index

    pb_mask_i = batch["template_pseudo_beta_mask"][..., ti, :]  # [B, N]
    pb_pair_mask = pb_mask_i[..., None] * pb_mask_i[..., None, :]  # [B, N, N]

    bb_mask_i = batch["template_backbone_frame_mask"][..., ti, :]  # [B, N]
    bb_pair_mask = bb_mask_i[..., None] * bb_mask_i[..., None, :]  # [B, N, N]

    asym = batch["asym_id"]  # [B, N]
    multichain = (asym[..., None] == asym[..., None, :]).to(dtype)  # [B, N, N]

    pb_pair_mask = (pb_pair_mask * multichain).to(dtype)[..., None]
    bb_pair_mask = (bb_pair_mask * multichain).to(dtype)[..., None]

    unit_vec = batch["template_unit_vector"][..., ti, :, :, :]  # [B, N, N, 3]
    if unit_vec.dtype != dtype:
        unit_vec = unit_vec.to(dtype)

    return torch.cat([pb_pair_mask, unit_vec, bb_pair_mask], dim=-1)


def _fetch_template_slice(
    batch: dict, template_index: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dgram, restype)`` for one template as contiguous [B, N, ...]."""
    dg = batch["template_distogram"][..., template_index, :, :, :]  # [B, N, N, C_DG]
    if dg.device.type != "cuda":
        dg = dg.to("cuda", non_blocking=True)
    if dg.dtype != dtype:
        dg = dg.to(dtype)

    re = batch["template_restype"][..., template_index, :, :]  # [B, N, C_RE]
    if re.device.type != "cuda":
        re = re.to("cuda", non_blocking=True)
    if re.dtype != dtype:
        re = re.to(dtype)
    return dg.contiguous(), re.contiguous()


# Rows-per-chunk when materializing LN(z). A 64-row chunk at N=1264 fp32
# is 41 MiB (0.05U), amortized to 0 across the loop as it rises and falls.
_LN_CHUNK_ROWS = 64


def _chunked_ln_linear_z(
    z: torch.Tensor,
    module: nn.Module,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``LN(z) @ W_z.T + B_z`` writing directly into a fresh output.

    Chunks along the row axis so the LN intermediate stays at
    ``chunk_rows × N × c_z`` (~0.05U at chunk=64 for N=1264 fp32) instead
    of the full 0.5U ``z_projected`` transient the eager path materializes.
    """
    B, N, N2, c_z = z.shape
    c_out = module.linear_z.weight.shape[0]
    out = torch.empty(B, N, N, c_out, dtype=dtype, device=z.device)

    ln_w = module.layer_norm_z.weight
    ln_b = module.layer_norm_z.bias
    lin_w = module.linear_z.weight
    lin_b = module.linear_z.bias
    if ln_w.dtype != dtype:
        ln_w = ln_w.to(dtype)
    if ln_b is not None and ln_b.dtype != dtype:
        ln_b = ln_b.to(dtype)
    if lin_w.dtype != dtype:
        lin_w = lin_w.to(dtype)
    if lin_b is not None and lin_b.dtype != dtype:
        lin_b = lin_b.to(dtype)

    # Chunk sizing: at small N the fixed 64 rows becomes a large fraction
    # of 1U; scale down so the LN scratch stays well below 0.1U.  1U bytes
    # ∝ N² × c_z × elem, chunk bytes ∝ chunk × N × c_z × elem, so the
    # ratio is chunk / N.  Cap at 0.1: chunk <= max(16, N // 10).
    chunk_rows = min(_LN_CHUNK_ROWS, max(16, N // 10))

    for start in range(0, N, chunk_rows):
        end = min(N, start + chunk_rows)
        z_chunk = z[:, start:end].contiguous()
        z_norm = F.layer_norm(
            z_chunk, (c_z,), ln_w, ln_b, module.layer_norm_z.eps
        )
        # Write straight into the output slice — no persistent linear buffer.
        out[:, start:end] = F.linear(z_norm, lin_w, lin_b)
        del z_chunk, z_norm
    return out


def _build_scalar_weight(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Concat the 5 scalar-feature Linear weights into a single ``[c_out, 5]``.

    Matches the eager ``_linear_scalar_stack`` weight ordering
    (pseudo_beta_mask, x, y, z, backbone_mask).
    """
    return torch.cat(
        (
            module.pseudo_beta_mask_linear.weight,
            module.x_linear.weight,
            module.y_linear.weight,
            module.z_linear.weight,
            module.backbone_mask_linear.weight,
        ),
        dim=-1,
    ).to(dtype=dtype)


def _add_restype_projections_(
    a: torch.Tensor,
    restype: torch.Tensor,
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Project O(N) restype inputs and broadcast-add without pair expansion."""
    for linear, dim in (
        (module.aatype_linear_1, -2),
        (module.aatype_linear_2, -3),
    ):
        weight = linear.weight.to(dtype=dtype)
        a.add_(F.linear(restype, weight, None).unsqueeze(dim))


def _fetch_coordinate_template_slice(
    batch: dict,
    template_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch compact coordinates, masks, and restype for one template."""
    values = (
        batch["template_pseudo_beta_coords"][..., template_index, :, :],
        batch["template_frame_atom_coords"][..., template_index, :, :, :],
        batch["template_pseudo_beta_mask"][..., template_index, :],
        batch["template_backbone_frame_mask"][..., template_index, :],
        batch["template_restype"][..., template_index, :, :],
    )
    return tuple(
        value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        for value in values
    )


def template_coordinate_projection_add_reference_(
    out: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    chunk_rows: int = 32,
) -> None:
    """Chunked reference for coordinate construction and direct projection."""
    n_token = out.shape[-2]
    lower = torch.linspace(
        3.25, 50.75, 39, device=out.device, dtype=torch.float32
    ).square()
    dgram_lookup = dgram_weight.to(dtype=out.dtype).transpose(0, 1)
    scalar_weight = scalar_weight.to(dtype=out.dtype)

    n_coords = frame_atom_coords[..., 0, :]
    ca_coords = frame_atom_coords[..., 1, :]
    c_coords = frame_atom_coords[..., 2, :]

    axis_x = F.normalize(c_coords - ca_coords, dim=-1, eps=1e-6)
    axis_y = n_coords - ca_coords
    axis_y = axis_y - (axis_y * axis_x).sum(dim=-1, keepdim=True) * axis_x
    axis_y = F.normalize(axis_y, dim=-1, eps=1e-6)
    axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)

    for start in range(0, n_token, chunk_rows):
        stop = min(start + chunk_rows, n_token)
        pb_delta = (
            pseudo_beta_coords[:, start:stop, None, :]
            - pseudo_beta_coords[:, None, :, :]
        )
        dist2 = pb_delta.square().sum(dim=-1)
        bin_index = torch.searchsorted(lower, dist2, right=False) - 1
        safe_bin = bin_index.clamp(min=0, max=38)
        bin_lower = lower[safe_bin]
        bin_upper = torch.where(
            safe_bin == 38,
            torch.full_like(bin_lower, 1.0e8),
            lower[(safe_bin + 1).clamp(max=38)],
        )
        in_bin = (bin_index >= 0) & (dist2 > bin_lower) & (dist2 < bin_upper)

        same_chain = (asym_id[:, start:stop, None] == asym_id[:, None, :]).to(out.dtype)
        pb_pair_mask = (
            pseudo_beta_mask[:, start:stop, None]
            * pseudo_beta_mask[:, None, :]
            * same_chain
        )
        bb_pair_mask = (
            backbone_frame_mask[:, start:stop, None]
            * backbone_frame_mask[:, None, :]
            * same_chain
        )

        dgram_projection = F.embedding(safe_bin, dgram_lookup)
        dgram_projection.mul_((in_bin.to(out.dtype) * pb_pair_mask)[..., None])
        out[:, start:stop].add_(dgram_projection)

        delta = ca_coords[:, None, :, :] - ca_coords[:, start:stop, None, :]
        local = torch.stack(
            (
                (delta * axis_x[:, start:stop, None, :]).sum(dim=-1),
                (delta * axis_y[:, start:stop, None, :]).sum(dim=-1),
                (delta * axis_z[:, start:stop, None, :]).sum(dim=-1),
            ),
            dim=-1,
        )
        local = F.normalize(local, dim=-1, eps=1e-6)
        local.mul_(bb_pair_mask[..., None])
        scalar_features = torch.cat(
            (pb_pair_mask[..., None], local, bb_pair_mask[..., None]), dim=-1
        )
        out[:, start:stop].add_(
            F.linear(scalar_features.to(out.dtype), scalar_weight, None)
        )


def fused_template_coordinate_pair_embedder_inference(
    module: nn.Module,
    batch: dict,
    z: torch.Tensor,
    template_index: int,
) -> torch.Tensor:
    """Embed one compact-coordinate template without raw pairwise features."""
    if module.training or torch.is_grad_enabled():
        raise RuntimeError(
            "fused_template_coordinate_pair_embedder_inference is inference-only"
        )
    if z.dim() != 4 or z.shape[0] != 1 or z.shape[-3] != z.shape[-2]:
        raise ValueError("Coordinate template embedding requires z [1,N,N,C]")

    dtype = z.dtype
    a = _chunked_ln_linear_z(z, module, dtype)
    (
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        restype,
    ) = _fetch_coordinate_template_slice(
        batch, template_index, dtype=dtype, device=z.device
    )
    _add_restype_projections_(a, restype, module, dtype)

    dgram_weight = module.dgram_linear.weight
    scalar_weight = _build_scalar_weight(module, dtype)
    asym_id = batch["asym_id"].to(device=z.device, non_blocking=True).contiguous()
    use_triton = (
        is_fused_template_pair_embedder_enabled()
        and z.is_cuda
        and dtype == torch.float32
        and a.shape[-1] == 64
        and dgram_weight.shape == (64, 39)
        and scalar_weight.shape == (64, 5)
    )
    if use_triton:
        template_coordinate_projection_add_(
            a,
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
            dgram_weight.contiguous(),
            scalar_weight.contiguous(),
        )
    else:
        template_coordinate_projection_add_reference_(
            a,
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
            dgram_weight,
            scalar_weight,
        )
    return a.unsqueeze(-4)


def fused_template_pair_embedder_inference(
    module: nn.Module,
    batch: dict,
    z: torch.Tensor,
    template_index: int,
) -> torch.Tensor:
    """Compute ``[B, 1, N, N, c_t]`` for one template with fused primitives.

    The output tensor ``a`` is allocated exactly once (by
    ``fused_ln_linear_inference``); subsequent projections accumulate into
    ``a`` via ``torch.addmm`` and broadcast ``add_``.

    Args:
        module: a ``TemplatePairEmbedderAllAtom`` supplying weights + LN.
        batch: full multi-template feature dict (same keys/shape as eager).
        z: pair representation ``[B, N, N, c_z]`` — contiguous.
        template_index: which template to embed.

    Returns:
        Tensor ``[B, 1, N, N, c_t]`` matching the eager path's output.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("fused_template_pair_embedder_inference is inference-only")

    dtype = batch["template_unit_vector"].dtype
    B, N, N2, c_z = z.shape
    assert N == N2
    c_out = module.linear_z.weight.shape[0]

    # ── LN → matmul on z, chunked: allocates ``a`` (0.5U) directly. ──
    # LN scratch stays at ~0.05U per iteration (rises and falls); no
    # persistent 0.5U ``z_projected`` intermediate.
    a = _chunked_ln_linear_z(z, module, dtype)  # [B, N, N, c_out]

    # ── dgram: in-place addmm into ``a``. ──
    dgram, restype = _fetch_template_slice(batch, template_index, dtype)
    a_flat = a.reshape(-1, c_out)
    dg_flat = dgram.reshape(-1, dgram.shape[-1])
    w_dg = module.dgram_linear.weight
    if w_dg.dtype != dtype:
        w_dg = w_dg.to(dtype)
    torch.addmm(a_flat, dg_flat, w_dg.t(), out=a_flat)
    # Release the [B, N, N, c_dgram] slice (~0.30U at c_dgram=39, N=1264)
    # immediately — nothing below this line references dgram.
    del dg_flat, dgram

    # ── aatype_i: [B, N, C_RE] @ [C_OUT, C_RE].T -> [B, N, C_OUT]; ──
    # broadcast-add along the j axis.  Zero materialization of
    # ``restype.expand(..., j, ...).to(dtype)``.
    _add_restype_projections_(a, restype, module, dtype)
    del restype

    # ── scalar_feats: cat 5-channel small tensor, in-place addmm. ──
    scalar = _build_scalar_feats(batch, template_index, dtype)  # [B, N, N, 5]
    w_sc = _build_scalar_weight(module, dtype)
    sc_flat = scalar.reshape(-1, scalar.shape[-1])
    torch.addmm(a_flat, sc_flat, w_sc.t(), out=a_flat)
    del sc_flat, scalar

    return a.unsqueeze(-4)  # add template dim
