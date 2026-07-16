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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: length-generic Triton template
# coordinate feature construction and fused pair projection.

"""Coordinate-derived template pair projection.

The fixed-shape projection (39 distogram inputs and five scalar inputs into
64 output channels) is fused with coordinate-to-pair feature construction.
No pairwise feature tensor is written to HBM. Sequence length and all tensor
strides remain runtime values so one compiled kernel serves every length.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    _BLOCK_PAIRS = 16
    _BLOCK_CHANNELS = 64

    @triton.jit(
        do_not_specialize=[
            "N",
            "stride_out_i",
            "stride_out_j",
            "stride_out_c",
            "stride_pb_i",
            "stride_pb_xyz",
            "stride_frame_i",
            "stride_frame_atom",
            "stride_frame_xyz",
            "stride_pb_mask_i",
            "stride_bb_mask_i",
            "stride_asym_i",
            "stride_w_dgram_c",
            "stride_w_dgram_bin",
            "stride_w_scalar_c",
            "stride_w_scalar_feat",
        ],
        do_not_specialize_on_alignment=[
            "out_ptr",
            "pb_coords_ptr",
            "frame_coords_ptr",
            "pb_mask_ptr",
            "bb_mask_ptr",
            "asym_id_ptr",
            "w_dgram_ptr",
            "w_scalar_ptr",
        ],
    )
    def _template_coordinate_projection_kernel(
        out_ptr,
        pb_coords_ptr,
        frame_coords_ptr,
        pb_mask_ptr,
        bb_mask_ptr,
        asym_id_ptr,
        w_dgram_ptr,
        w_scalar_ptr,
        N,
        stride_out_i,
        stride_out_j,
        stride_out_c,
        stride_pb_i,
        stride_pb_xyz,
        stride_frame_i,
        stride_frame_atom,
        stride_frame_xyz,
        stride_pb_mask_i,
        stride_bb_mask_i,
        stride_asym_i,
        stride_w_dgram_c,
        stride_w_dgram_bin,
        stride_w_scalar_c,
        stride_w_scalar_feat,
        BLOCK_PAIRS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        channel = tl.arange(0, BLOCK_CHANNELS)
        pair_mask = pair < N * N

        i = pair // N
        j = pair - i * N

        pb_ix = tl.load(
            pb_coords_ptr + i * stride_pb_i,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_iy = tl.load(
            pb_coords_ptr + i * stride_pb_i + stride_pb_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_iz = tl.load(
            pb_coords_ptr + i * stride_pb_i + 2 * stride_pb_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_jx = tl.load(
            pb_coords_ptr + j * stride_pb_i,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_jy = tl.load(
            pb_coords_ptr + j * stride_pb_i + stride_pb_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_jz = tl.load(
            pb_coords_ptr + j * stride_pb_i + 2 * stride_pb_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pb_dx = pb_ix - pb_jx
        pb_dy = pb_iy - pb_jy
        pb_dz = pb_iz - pb_jz
        dist2 = pb_dx * pb_dx + pb_dy * pb_dy + pb_dz * pb_dz

        # Current template distogram uses 39 strict open intervals with
        # linearly spaced (unsquared) edges from 3.25 to 50.75 Angstrom.
        min_bin = 3.25
        bin_step = 1.25
        distance = tl.sqrt(dist2)
        bin_index = tl.floor((distance - min_bin) / bin_step).to(tl.int32)
        bin_index = tl.maximum(0, tl.minimum(38, bin_index))
        lower = min_bin + bin_index.to(tl.float32) * bin_step
        lower2 = lower * lower
        upper = lower + bin_step
        upper2 = tl.where(bin_index == 38, 1.0e8, upper * upper)
        in_bin = (dist2 > lower2) & (dist2 < upper2)

        pb_mask_i = tl.load(
            pb_mask_ptr + i * stride_pb_mask_i, mask=pair_mask, other=0.0
        ).to(tl.float32)
        pb_mask_j = tl.load(
            pb_mask_ptr + j * stride_pb_mask_i, mask=pair_mask, other=0.0
        ).to(tl.float32)
        bb_mask_i = tl.load(
            bb_mask_ptr + i * stride_bb_mask_i, mask=pair_mask, other=0.0
        ).to(tl.float32)
        bb_mask_j = tl.load(
            bb_mask_ptr + j * stride_bb_mask_i, mask=pair_mask, other=0.0
        ).to(tl.float32)
        asym_i = tl.load(asym_id_ptr + i * stride_asym_i, mask=pair_mask, other=0)
        asym_j = tl.load(asym_id_ptr + j * stride_asym_i, mask=pair_mask, other=1)
        same_chain = (asym_i == asym_j).to(tl.float32)
        pb_pair_mask = pb_mask_i * pb_mask_j * same_chain
        bb_pair_mask = bb_mask_i * bb_mask_j * same_chain

        frame_i_base = frame_coords_ptr + i * stride_frame_i
        n_x = tl.load(frame_i_base, mask=pair_mask, other=0.0).to(tl.float32)
        n_y = tl.load(frame_i_base + stride_frame_xyz, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        n_z = tl.load(
            frame_i_base + 2 * stride_frame_xyz, mask=pair_mask, other=0.0
        ).to(tl.float32)
        ca_x = tl.load(frame_i_base + stride_frame_atom, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        ca_y = tl.load(
            frame_i_base + stride_frame_atom + stride_frame_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        ca_z = tl.load(
            frame_i_base + stride_frame_atom + 2 * stride_frame_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        c_x = tl.load(
            frame_i_base + 2 * stride_frame_atom, mask=pair_mask, other=0.0
        ).to(tl.float32)
        c_y = tl.load(
            frame_i_base + 2 * stride_frame_atom + stride_frame_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        c_z = tl.load(
            frame_i_base + 2 * stride_frame_atom + 2 * stride_frame_xyz,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        frame_j_base = frame_coords_ptr + j * stride_frame_i + stride_frame_atom
        j_ca_x = tl.load(frame_j_base, mask=pair_mask, other=0.0).to(tl.float32)
        j_ca_y = tl.load(frame_j_base + stride_frame_xyz, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        j_ca_z = tl.load(
            frame_j_base + 2 * stride_frame_xyz, mask=pair_mask, other=0.0
        ).to(tl.float32)

        axis_x_x = c_x - ca_x
        axis_x_y = c_y - ca_y
        axis_x_z = c_z - ca_z
        inv_x_norm = tl.rsqrt(
            tl.maximum(
                axis_x_x * axis_x_x + axis_x_y * axis_x_y + axis_x_z * axis_x_z,
                1.0e-12,
            )
        )
        axis_x_x *= inv_x_norm
        axis_x_y *= inv_x_norm
        axis_x_z *= inv_x_norm

        axis_y_x = n_x - ca_x
        axis_y_y = n_y - ca_y
        axis_y_z = n_z - ca_z
        projection = axis_y_x * axis_x_x + axis_y_y * axis_x_y + axis_y_z * axis_x_z
        axis_y_x -= projection * axis_x_x
        axis_y_y -= projection * axis_x_y
        axis_y_z -= projection * axis_x_z
        inv_y_norm = tl.rsqrt(
            tl.maximum(
                axis_y_x * axis_y_x + axis_y_y * axis_y_y + axis_y_z * axis_y_z,
                1.0e-12,
            )
        )
        axis_y_x *= inv_y_norm
        axis_y_y *= inv_y_norm
        axis_y_z *= inv_y_norm

        axis_z_x = axis_x_y * axis_y_z - axis_x_z * axis_y_y
        axis_z_y = axis_x_z * axis_y_x - axis_x_x * axis_y_z
        axis_z_z = axis_x_x * axis_y_y - axis_x_y * axis_y_x

        delta_x = j_ca_x - ca_x
        delta_y = j_ca_y - ca_y
        delta_z = j_ca_z - ca_z
        local_x = delta_x * axis_x_x + delta_y * axis_x_y + delta_z * axis_x_z
        local_y = delta_x * axis_y_x + delta_y * axis_y_y + delta_z * axis_y_z
        local_z = delta_x * axis_z_x + delta_y * axis_z_y + delta_z * axis_z_z
        inv_delta_norm = tl.rsqrt(
            tl.maximum(
                local_x * local_x + local_y * local_y + local_z * local_z,
                1.0e-12,
            )
        )
        local_x *= inv_delta_norm
        local_y *= inv_delta_norm
        local_z *= inv_delta_norm

        out_offsets = (
            i[:, None] * stride_out_i
            + j[:, None] * stride_out_j
            + channel[None, :] * stride_out_c
        )
        out = tl.load(out_ptr + out_offsets, mask=pair_mask[:, None]).to(tl.float32)

        w_dgram = tl.load(
            w_dgram_ptr
            + channel[None, :] * stride_w_dgram_c
            + bin_index[:, None] * stride_w_dgram_bin,
            mask=pair_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        out += w_dgram * (pb_pair_mask * in_bin.to(tl.float32))[:, None]

        w_pb = tl.load(
            w_scalar_ptr + channel * stride_w_scalar_c + 0 * stride_w_scalar_feat
        ).to(tl.float32)
        w_x = tl.load(
            w_scalar_ptr + channel * stride_w_scalar_c + 1 * stride_w_scalar_feat
        ).to(tl.float32)
        w_y = tl.load(
            w_scalar_ptr + channel * stride_w_scalar_c + 2 * stride_w_scalar_feat
        ).to(tl.float32)
        w_z = tl.load(
            w_scalar_ptr + channel * stride_w_scalar_c + 3 * stride_w_scalar_feat
        ).to(tl.float32)
        w_bb = tl.load(
            w_scalar_ptr + channel * stride_w_scalar_c + 4 * stride_w_scalar_feat
        ).to(tl.float32)
        out += pb_pair_mask[:, None] * w_pb[None, :]
        out += bb_pair_mask[:, None] * (
            local_x[:, None] * w_x[None, :]
            + local_y[:, None] * w_y[None, :]
            + local_z[:, None] * w_z[None, :]
            + w_bb[None, :]
        )

        tl.store(out_ptr + out_offsets, out, mask=pair_mask[:, None])


def template_coordinate_projection_add_(
    out: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
) -> None:
    """Add coordinate-derived template projections to ``out`` in place.

    This launcher intentionally supports only the production inference shape:
    B=1, fp32, 39 distogram inputs, five scalar inputs, and 64 output channels.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for coordinate template projection")
    if not (
        out.is_cuda
        and out.dtype == torch.float32
        and out.shape[0] == 1
        and out.shape[-1] == 64
        and dgram_weight.shape == (64, 39)
        and scalar_weight.shape == (64, 5)
    ):
        raise ValueError("Unsupported coordinate template projection configuration")

    n_token = out.shape[1]
    if out.shape != (1, n_token, n_token, 64):
        raise ValueError(f"Expected out [1,N,N,64], got {tuple(out.shape)}")

    pb = pseudo_beta_coords[0]
    frame = frame_atom_coords[0]
    pb_mask = pseudo_beta_mask[0]
    bb_mask = backbone_frame_mask[0]
    asym = asym_id[0]
    grid = (triton.cdiv(n_token * n_token, _BLOCK_PAIRS),)
    _template_coordinate_projection_kernel[grid](
        out,
        pb,
        frame,
        pb_mask,
        bb_mask,
        asym,
        dgram_weight,
        scalar_weight,
        n_token,
        out.stride(1),
        out.stride(2),
        out.stride(3),
        pb.stride(0),
        pb.stride(1),
        frame.stride(0),
        frame.stride(1),
        frame.stride(2),
        pb_mask.stride(0),
        bb_mask.stride(0),
        asym.stride(0),
        dgram_weight.stride(0),
        dgram_weight.stride(1),
        scalar_weight.stride(0),
        scalar_weight.stride(1),
        BLOCK_PAIRS=_BLOCK_PAIRS,
        BLOCK_CHANNELS=_BLOCK_CHANNELS,
        num_warps=4,
        num_stages=1,
    )
