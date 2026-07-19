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

"""Inference dispatcher for online template coordinate pair features.

Builds one template's pair embedding from compact O(N) coordinates. On CUDA
fp32 with Triton available, projection uses the length-generic kernel in
``fused_template_coordinate`` (N and strides are runtime values). Otherwise
falls back to a chunked eager reference with the same math.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_add_,
    )

    _TRITON_KERNEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_KERNEL_AVAILABLE = False


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val in {"1", "true", "yes", "on"}


def is_fused_template_coordinate_enabled() -> bool:
    """Runtime gate for the Triton coordinate projection kernel."""
    if not _TRITON_KERNEL_AVAILABLE:
        return False
    return _env_flag("OPENFOLD3_FUSED_TEMPLATE_COORD", "1")


_LN_CHUNK_ROWS = 64


def _chunked_ln_linear_z(
    z: torch.Tensor,
    module: nn.Module,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``LN(z) @ W_z.T + B_z`` writing directly into a fresh output."""
    B, N, _, c_z = z.shape
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

    chunk_rows = min(_LN_CHUNK_ROWS, max(16, N // 10))
    for start in range(0, N, chunk_rows):
        end = min(N, start + chunk_rows)
        z_chunk = z[:, start:end].contiguous()
        z_norm = F.layer_norm(
            z_chunk, (c_z,), ln_w, ln_b, module.layer_norm_z.eps
        )
        out[:, start:end] = F.linear(z_norm, lin_w, lin_b)
        del z_chunk, z_norm
    return out


def _build_scalar_weight(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Concat scalar-feature Linear weights into ``[c_out, 5]``."""
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
        is_fused_template_coordinate_enabled()
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
