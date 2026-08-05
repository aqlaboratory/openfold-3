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

"""Online template coordinate pair features with guarded training support.

Builds one template's pair embedding from compact O(N) coordinates. On CUDA
fp32/bf16 with Triton available, projection uses the length-generic kernel in
``fused_template_coordinate`` (N and strides are runtime values; accumulate in
fp32). Otherwise falls back to a chunked eager reference with the same math.
Training computes model-parameter gradients without retaining pairwise
coordinate features.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection,
        template_coordinate_projection_add_,
        template_coordinate_projection_backward,
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
        z_norm = F.layer_norm(z_chunk, (c_z,), ln_w, ln_b, module.layer_norm_z.eps)
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


def _add_restype_projections(
    a: torch.Tensor,
    restype: torch.Tensor,
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
) -> torch.Tensor:
    """Project O(N) restype inputs and broadcast-add without pair expansion."""
    for linear, dim in (
        (module.aatype_linear_1, -2),
        (module.aatype_linear_2, -3),
    ):
        weight = linear.weight.to(dtype=dtype)
        update = F.linear(restype, weight, None).unsqueeze(dim)
        if inplace:
            a.add_(update)
        else:
            a = a + update
    return a


def _fetch_coordinate_template_slice(
    batch: dict,
    template_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch fp32 geometry/masks and activation-dtype restype for one template."""
    return (
        batch["template_pseudo_beta_coords"][..., template_index, :, :]
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .contiguous(),
        batch["template_frame_atom_coords"][..., template_index, :, :, :]
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .contiguous(),
        batch["template_pseudo_beta_mask"][..., template_index, :]
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .contiguous(),
        batch["template_backbone_frame_mask"][..., template_index, :]
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .contiguous(),
        batch["template_restype"][..., template_index, :, :]
        .to(device=device, dtype=dtype, non_blocking=True)
        .contiguous(),
    )


def _coordinate_frame_axes(
    frame_atom_coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_atom_coords = frame_atom_coords.to(torch.float32)
    n_coords = frame_atom_coords[..., 0, :]
    ca_coords = frame_atom_coords[..., 1, :]
    c_coords = frame_atom_coords[..., 2, :]
    axis_x = F.normalize(c_coords - ca_coords, dim=-1, eps=1e-6)
    axis_y = n_coords - ca_coords
    axis_y = axis_y - (axis_y * axis_x).sum(dim=-1, keepdim=True) * axis_x
    axis_y = F.normalize(axis_y, dim=-1, eps=1e-6)
    axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)
    return ca_coords, axis_x, axis_y, axis_z


def _coordinate_feature_chunk(
    pseudo_beta_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    ca_coords: torch.Tensor,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    axis_z: torch.Tensor,
    lower: torch.Tensor,
    start: int,
    stop: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pb_coords = pseudo_beta_coords.to(torch.float32)
    pb_delta = pb_coords[:, start:stop, None, :] - pb_coords[:, None, :, :]
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

    same_chain = (asym_id[:, start:stop, None] == asym_id[:, None, :]).to(dtype)
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
    local = local * bb_pair_mask[..., None]
    scalar_features = torch.cat(
        (pb_pair_mask[..., None], local, bb_pair_mask[..., None]), dim=-1
    )
    dgram_scale = in_bin.to(dtype) * pb_pair_mask
    return safe_bin, dgram_scale, scalar_features


def _coordinate_projection_reference_loop(
    source: torch.Tensor,
    out: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    chunk_rows: int,
    *,
    inplace: bool,
) -> torch.Tensor:
    """Row-chunked eager projection into ``out`` (in-place add or fresh write)."""
    n_token = source.shape[-2]
    lower = torch.linspace(
        3.25, 50.75, 39, device=source.device, dtype=torch.float32
    ).square()
    dgram_lookup = dgram_weight.to(dtype=source.dtype).transpose(0, 1)
    scalar_weight = scalar_weight.to(dtype=source.dtype)
    ca_coords, axis_x, axis_y, axis_z = _coordinate_frame_axes(frame_atom_coords)
    for start in range(0, n_token, chunk_rows):
        stop = min(start + chunk_rows, n_token)
        safe_bin, dgram_scale, scalar_features = _coordinate_feature_chunk(
            pseudo_beta_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
            ca_coords,
            axis_x,
            axis_y,
            axis_z,
            lower,
            start,
            stop,
            source.dtype,
        )
        dgram_projection = F.embedding(safe_bin, dgram_lookup)
        dgram_projection = dgram_projection * dgram_scale[..., None]
        scalar_projection = F.linear(
            scalar_features.to(source.dtype), scalar_weight, None
        )
        if inplace:
            out[:, start:stop].add_(dgram_projection)
            out[:, start:stop].add_(scalar_projection)
        else:
            out[:, start:stop] = (
                source[:, start:stop] + dgram_projection + scalar_projection
            )
    return out


def template_coordinate_projection_reference(
    source: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    chunk_rows: int = 32,
) -> torch.Tensor:
    """Differentiable row-chunked coordinate projection with a fresh output."""
    return _coordinate_projection_reference_loop(
        source,
        torch.empty_like(source),
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
        chunk_rows,
        inplace=False,
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
    """Inference-only in-place eager reference projection."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "In-place template coordinate reference requires disabled grad mode"
        )
    _coordinate_projection_reference_loop(
        out,
        out,
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
        chunk_rows,
        inplace=True,
    )


class _TemplateCoordinateProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        pseudo_beta_coords: torch.Tensor,
        frame_atom_coords: torch.Tensor,
        pseudo_beta_mask: torch.Tensor,
        backbone_frame_mask: torch.Tensor,
        asym_id: torch.Tensor,
        dgram_weight: torch.Tensor,
        scalar_weight: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
        )
        return template_coordinate_projection(
            source,
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
            dgram_weight,
            scalar_weight,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
        ) = ctx.saved_tensors
        needs_source = ctx.needs_input_grad[0]
        needs_dgram = ctx.needs_input_grad[6]
        needs_scalar = ctx.needs_input_grad[7]
        grad_dgram, grad_scalar = template_coordinate_projection_backward(
            grad_output,
            pseudo_beta_coords,
            frame_atom_coords,
            pseudo_beta_mask,
            backbone_frame_mask,
            asym_id,
            compute_dgram=needs_dgram,
            compute_scalar=needs_scalar,
        )

        return (
            grad_output if needs_source else None,
            None,
            None,
            None,
            None,
            None,
            grad_dgram,
            grad_scalar,
        )


_FUSED_AWAY_LINEAR_NAMES = (
    "dgram_linear",
    "aatype_linear_1",
    "aatype_linear_2",
    "pseudo_beta_mask_linear",
    "x_linear",
    "y_linear",
    "z_linear",
    "backbone_mask_linear",
)


def _validate_fused_away_biases(module: nn.Module) -> None:
    biased = [
        name
        for name in _FUSED_AWAY_LINEAR_NAMES
        if getattr(module, name).bias is not None
    ]
    if biased:
        raise ValueError(
            "Coordinate template projection requires bias-free feature linears; "
            f"found biases in {', '.join(biased)}"
        )


def _can_use_triton(
    source: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
) -> bool:
    inputs = (
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
    )
    return (
        is_fused_template_coordinate_enabled()
        and source.is_cuda
        and source.dtype in (torch.float32, torch.bfloat16)
        and source.shape[0] == 1
        and source.shape[-1] == 64
        and dgram_weight.shape == (64, 39)
        and scalar_weight.shape == (64, 5)
        and all(x.is_cuda and x.device == source.device for x in inputs)
    )


def fused_template_coordinate_pair_embedder(
    module: nn.Module,
    batch: dict,
    z: torch.Tensor,
    template_index: int,
) -> torch.Tensor:
    """Embed one compact-coordinate template with inference/training dispatch."""
    if z.dim() != 4 or z.shape[-3] != z.shape[-2]:
        raise ValueError("Coordinate template embedding requires z [B,N,N,C]")
    _validate_fused_away_biases(module)

    dtype = z.dtype
    inference_inplace = not module.training and not torch.is_grad_enabled()
    if torch.is_grad_enabled():
        a = module.linear_z(module.layer_norm_z(z))
    else:
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
    coordinate_tensors = (
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
    )
    if any(x.requires_grad for x in coordinate_tensors):
        raise ValueError("Template coordinate inputs are non-differentiable data")
    a = _add_restype_projections(a, restype, module, dtype, inplace=inference_inplace)

    asym_id = batch["asym_id"].to(device=z.device, non_blocking=True).contiguous()
    dgram_weight = module.dgram_linear.weight
    scalar_weight = _build_scalar_weight(module, torch.float32)
    use_triton = _can_use_triton(
        a,
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
    )
    if use_triton:
        dgram_weight = dgram_weight.float().contiguous()
        scalar_weight = scalar_weight.contiguous()
        with torch.autocast(device_type="cuda", enabled=False):
            if torch.is_grad_enabled():
                a = _TemplateCoordinateProjectionFunction.apply(
                    a,
                    pseudo_beta_coords,
                    frame_atom_coords,
                    pseudo_beta_mask,
                    backbone_frame_mask,
                    asym_id,
                    dgram_weight,
                    scalar_weight,
                )
            elif inference_inplace:
                template_coordinate_projection_add_(
                    a,
                    pseudo_beta_coords,
                    frame_atom_coords,
                    pseudo_beta_mask,
                    backbone_frame_mask,
                    asym_id,
                    dgram_weight,
                    scalar_weight,
                )
            else:
                a = template_coordinate_projection(
                    a,
                    pseudo_beta_coords,
                    frame_atom_coords,
                    pseudo_beta_mask,
                    backbone_frame_mask,
                    asym_id,
                    dgram_weight,
                    scalar_weight,
                )
    else:
        dgram_weight = dgram_weight.to(dtype=dtype)
        scalar_weight = scalar_weight.to(dtype=dtype)
        if inference_inplace:
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
        else:
            a = template_coordinate_projection_reference(
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
