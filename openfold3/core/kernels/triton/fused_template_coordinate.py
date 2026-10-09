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
Supports fp32/bf16 pair activations with fp32 accumulate.
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
    _BWD_BLOCK_PAIRS = 32
    _BWD_BLOCK_CHANNELS = 64
    _BWD_BLOCK_FEATURES = 16
    _BWD_SPLIT_K = 64
    _BWD_FEATURES = 44

    # Feature layout: [0, 39) distogram one-hot, 39 pseudo-beta pair mask,
    # 40..42 local unit vector xyz, 43 backbone pair mask.
    _C_OUT = tl.constexpr(64)
    _NUM_BINS = tl.constexpr(39)
    _NUM_FEATURES = tl.constexpr(_BWD_FEATURES)
    _MIN_BIN = tl.constexpr(3.25)
    _BIN_STEP = tl.constexpr(1.25)
    _NORM_EPS2 = tl.constexpr(1.0e-12)  # squared F.normalize(eps=1e-6)

    @triton.jit
    def _load_xyz(ptr, stride_xyz, mask):
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(ptr + stride_xyz, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(ptr + 2 * stride_xyz, mask=mask, other=0.0).to(tl.float32)
        return x, y, z

    @triton.jit
    def _normalize3(x, y, z):
        inv_norm = tl.rsqrt(tl.maximum(x * x + y * y + z * z, _NORM_EPS2))
        return x * inv_norm, y * inv_norm, z * inv_norm

    @triton.jit
    def _template_coordinate_pair_features(
        pair,
        pb_coords_ptr,
        frame_coords_ptr,
        pb_mask_ptr,
        bb_mask_ptr,
        asym_id_ptr,
        N,
        stride_pb_i,
        stride_pb_xyz,
        stride_frame_i,
        stride_frame_atom,
        stride_frame_xyz,
        stride_pb_mask_i,
        stride_bb_mask_i,
        stride_asym_i,
    ):
        # int64 so downstream i/j stride offsets cannot wrap: (N*N-1)*64
        # exceeds int32 at N>=5793 and caused CUDA illegal memory access.
        N64 = N.to(tl.int64)
        pair_mask = pair < N64 * N64
        i = pair // N64
        j = pair - i * N64

        pb_ix, pb_iy, pb_iz = _load_xyz(
            pb_coords_ptr + i * stride_pb_i, stride_pb_xyz, pair_mask
        )
        pb_jx, pb_jy, pb_jz = _load_xyz(
            pb_coords_ptr + j * stride_pb_i, stride_pb_xyz, pair_mask
        )
        pb_dx = pb_ix - pb_jx
        pb_dy = pb_iy - pb_jy
        pb_dz = pb_iz - pb_jz
        dist2 = pb_dx * pb_dx + pb_dy * pb_dy + pb_dz * pb_dz

        # Bin on squared edges with open intervals, as the reference. Rounding
        # cannot change the nearest edge, so compare dist2 to that edge only.
        k = tl.floor((tl.sqrt(dist2) - _MIN_BIN) / _BIN_STEP + 0.5).to(tl.int32)
        k = tl.maximum(0, tl.minimum(_NUM_BINS - 1, k))
        edge = _MIN_BIN + k.to(tl.float32) * _BIN_STEP
        edge2 = edge * edge
        bin_index = k - (dist2 < edge2).to(tl.int32)
        # 1e8 is the reference's squared upper edge of the last bin; it also
        # drops NaN distances, which fail every comparison.
        in_bin = (bin_index >= 0) & (dist2 != edge2) & (dist2 < 1.0e8)
        bin_index = tl.maximum(bin_index, 0)

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
        # Different fill values make masked tail lanes count as cross-chain.
        asym_i = tl.load(asym_id_ptr + i * stride_asym_i, mask=pair_mask, other=0)
        asym_j = tl.load(asym_id_ptr + j * stride_asym_i, mask=pair_mask, other=1)
        same_chain = (asym_i == asym_j).to(tl.float32)
        pb_pair_mask = pb_mask_i * pb_mask_j * same_chain
        bb_pair_mask = bb_mask_i * bb_mask_j * same_chain

        frame_i_base = frame_coords_ptr + i * stride_frame_i
        n_x, n_y, n_z = _load_xyz(frame_i_base, stride_frame_xyz, pair_mask)
        ca_x, ca_y, ca_z = _load_xyz(
            frame_i_base + stride_frame_atom, stride_frame_xyz, pair_mask
        )
        c_x, c_y, c_z = _load_xyz(
            frame_i_base + 2 * stride_frame_atom, stride_frame_xyz, pair_mask
        )
        j_ca_x, j_ca_y, j_ca_z = _load_xyz(
            frame_coords_ptr + j * stride_frame_i + stride_frame_atom,
            stride_frame_xyz,
            pair_mask,
        )

        axis_x_x, axis_x_y, axis_x_z = _normalize3(c_x - ca_x, c_y - ca_y, c_z - ca_z)

        axis_y_x = n_x - ca_x
        axis_y_y = n_y - ca_y
        axis_y_z = n_z - ca_z
        projection = axis_y_x * axis_x_x + axis_y_y * axis_x_y + axis_y_z * axis_x_z
        axis_y_x, axis_y_y, axis_y_z = _normalize3(
            axis_y_x - projection * axis_x_x,
            axis_y_y - projection * axis_x_y,
            axis_y_z - projection * axis_x_z,
        )

        axis_z_x = axis_x_y * axis_y_z - axis_x_z * axis_y_y
        axis_z_y = axis_x_z * axis_y_x - axis_x_x * axis_y_z
        axis_z_z = axis_x_x * axis_y_y - axis_x_y * axis_y_x

        delta_x = j_ca_x - ca_x
        delta_y = j_ca_y - ca_y
        delta_z = j_ca_z - ca_z
        local_x, local_y, local_z = _normalize3(
            delta_x * axis_x_x + delta_y * axis_x_y + delta_z * axis_x_z,
            delta_x * axis_y_x + delta_y * axis_y_y + delta_z * axis_y_z,
            delta_x * axis_z_x + delta_y * axis_z_y + delta_z * axis_z_z,
        )
        return (
            i,
            j,
            pair_mask,
            bin_index,
            in_bin,
            pb_pair_mask,
            bb_pair_mask,
            local_x,
            local_y,
            local_z,
        )

    @triton.jit(
        do_not_specialize=[
            "N",
            "stride_source_i",
            "stride_source_j",
            "stride_source_c",
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
            "source_ptr",
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
        source_ptr,
        out_ptr,
        pb_coords_ptr,
        frame_coords_ptr,
        pb_mask_ptr,
        bb_mask_ptr,
        asym_id_ptr,
        w_dgram_ptr,
        w_scalar_ptr,
        N,
        stride_source_i,
        stride_source_j,
        stride_source_c,
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

        (
            i,
            j,
            pair_mask,
            bin_index,
            in_bin,
            pb_pair_mask,
            bb_pair_mask,
            local_x,
            local_y,
            local_z,
        ) = _template_coordinate_pair_features(
            pair,
            pb_coords_ptr,
            frame_coords_ptr,
            pb_mask_ptr,
            bb_mask_ptr,
            asym_id_ptr,
            N,
            stride_pb_i,
            stride_pb_xyz,
            stride_frame_i,
            stride_frame_atom,
            stride_frame_xyz,
            stride_pb_mask_i,
            stride_bb_mask_i,
            stride_asym_i,
        )

        source_offsets = (
            i[:, None].to(tl.int64) * stride_source_i
            + j[:, None].to(tl.int64) * stride_source_j
            + channel[None, :].to(tl.int64) * stride_source_c
        )
        out_offsets = (
            i[:, None].to(tl.int64) * stride_out_i
            + j[:, None].to(tl.int64) * stride_out_j
            + channel[None, :].to(tl.int64) * stride_out_c
        )
        out = tl.load(source_ptr + source_offsets, mask=pair_mask[:, None]).to(
            tl.float32
        )

        w_dgram = tl.load(
            w_dgram_ptr
            + channel[None, :] * stride_w_dgram_c
            + bin_index[:, None] * stride_w_dgram_bin,
            mask=pair_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        out += w_dgram * (pb_pair_mask * in_bin.to(tl.float32))[:, None]

        w_scalar_base = w_scalar_ptr + channel * stride_w_scalar_c
        w_pb = tl.load(w_scalar_base).to(tl.float32)
        w_x = tl.load(w_scalar_base + stride_w_scalar_feat).to(tl.float32)
        w_y = tl.load(w_scalar_base + 2 * stride_w_scalar_feat).to(tl.float32)
        w_z = tl.load(w_scalar_base + 3 * stride_w_scalar_feat).to(tl.float32)
        w_bb = tl.load(w_scalar_base + 4 * stride_w_scalar_feat).to(tl.float32)
        out += pb_pair_mask[:, None] * w_pb[None, :]
        out += bb_pair_mask[:, None] * (
            local_x[:, None] * w_x[None, :]
            + local_y[:, None] * w_y[None, :]
            + local_z[:, None] * w_z[None, :]
            + w_bb[None, :]
        )

        tl.store(
            out_ptr + out_offsets,
            out.to(out_ptr.dtype.element_ty),
            mask=pair_mask[:, None],
        )

    @triton.jit(
        do_not_specialize=[
            "N",
            "stride_grad_i",
            "stride_grad_j",
            "stride_grad_c",
            "stride_pb_i",
            "stride_pb_xyz",
            "stride_frame_i",
            "stride_frame_atom",
            "stride_frame_xyz",
            "stride_pb_mask_i",
            "stride_bb_mask_i",
            "stride_asym_i",
            "stride_partial_split",
            "stride_partial_c",
            "stride_partial_f",
        ],
        do_not_specialize_on_alignment=[
            "grad_output_ptr",
            "pb_coords_ptr",
            "frame_coords_ptr",
            "pb_mask_ptr",
            "bb_mask_ptr",
            "asym_id_ptr",
            "partial_ptr",
        ],
    )
    def _template_coordinate_projection_bwd_partial_kernel(
        grad_output_ptr,
        pb_coords_ptr,
        frame_coords_ptr,
        pb_mask_ptr,
        bb_mask_ptr,
        asym_id_ptr,
        partial_ptr,
        N,
        stride_grad_i,
        stride_grad_j,
        stride_grad_c,
        stride_pb_i,
        stride_pb_xyz,
        stride_frame_i,
        stride_frame_atom,
        stride_frame_xyz,
        stride_pb_mask_i,
        stride_bb_mask_i,
        stride_asym_i,
        stride_partial_split,
        stride_partial_c,
        stride_partial_f,
        COMPUTE_DGRAM: tl.constexpr,
        COMPUTE_SCALAR: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        channel = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        feature = tl.program_id(1) * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
        split = tl.program_id(2)
        channel_mask = channel < _C_OUT
        feature_mask = feature < _NUM_FEATURES
        if not COMPUTE_DGRAM:
            feature_mask &= feature >= _NUM_BINS
        if not COMPUTE_SCALAR:
            feature_mask &= feature < _NUM_BINS

        accumulator = tl.zeros(
            (BLOCK_CHANNELS, BLOCK_FEATURES), dtype=tl.float32
        )
        channel_offsets = channel[None, :].to(tl.int64) * stride_grad_c
        pair_count = N.to(tl.int64) * N
        pairs_per_split = (
            pair_count + SPLIT_K * BLOCK_PAIRS - 1
        ) // (SPLIT_K * BLOCK_PAIRS) * BLOCK_PAIRS
        pair_start = split * pairs_per_split
        pair_stop = tl.minimum(pair_start + pairs_per_split, pair_count)
        while pair_start < pair_stop:
            pair = pair_start + tl.arange(0, BLOCK_PAIRS)
            (
                i,
                j,
                pair_mask,
                bin_index,
                in_bin,
                pb_pair_mask,
                bb_pair_mask,
                local_x,
                local_y,
                local_z,
            ) = _template_coordinate_pair_features(
                pair,
                pb_coords_ptr,
                frame_coords_ptr,
                pb_mask_ptr,
                bb_mask_ptr,
                asym_id_ptr,
                N,
                stride_pb_i,
                stride_pb_xyz,
                stride_frame_i,
                stride_frame_atom,
                stride_frame_xyz,
                stride_pb_mask_i,
                stride_bb_mask_i,
                stride_asym_i,
            )
            grad_offsets = (
                i[:, None].to(tl.int64) * stride_grad_i
                + j[:, None].to(tl.int64) * stride_grad_j
                + channel_offsets
            )
            grad = tl.load(
                grad_output_ptr + grad_offsets,
                mask=pair_mask[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # [BLOCK_PAIRS, BLOCK_FEATURES] slice of the 44-feature layout.
            f = feature[None, :]
            dgram_scale = pb_pair_mask * in_bin.to(tl.float32)
            features = tl.where(
                (f < _NUM_BINS) & (f == bin_index[:, None]),
                dgram_scale[:, None],
                0.0,
            )
            features += tl.where(f == _NUM_BINS, pb_pair_mask[:, None], 0.0)
            features += tl.where(
                f == _NUM_BINS + 1, (bb_pair_mask * local_x)[:, None], 0.0
            )
            features += tl.where(
                f == _NUM_BINS + 2, (bb_pair_mask * local_y)[:, None], 0.0
            )
            features += tl.where(
                f == _NUM_BINS + 3, (bb_pair_mask * local_z)[:, None], 0.0
            )
            features += tl.where(f == _NUM_BINS + 4, bb_pair_mask[:, None], 0.0)
            features = tl.where(
                pair_mask[:, None] & feature_mask[None, :], features, 0.0
            )
            accumulator = tl.dot(
                tl.trans(grad),
                features,
                accumulator,
                input_precision=INPUT_PRECISION,
                out_dtype=tl.float32,
            )
            pair_start += BLOCK_PAIRS

        partial_offsets = (
            split * stride_partial_split
            + channel[:, None] * stride_partial_c
            + feature[None, :] * stride_partial_f
        )
        tl.store(
            partial_ptr + partial_offsets,
            accumulator,
            mask=channel_mask[:, None] & feature_mask[None, :],
        )

    @triton.jit(
        do_not_specialize=[
            "stride_partial_split",
            "stride_partial_c",
            "stride_partial_f",
            "stride_dgram_c",
            "stride_dgram_f",
            "stride_scalar_c",
            "stride_scalar_f",
        ],
        do_not_specialize_on_alignment=[
            "partial_ptr",
            "grad_dgram_ptr",
            "grad_scalar_ptr",
        ],
    )
    def _template_coordinate_projection_bwd_reduce_kernel(
        partial_ptr,
        grad_dgram_ptr,
        grad_scalar_ptr,
        stride_partial_split,
        stride_partial_c,
        stride_partial_f,
        stride_dgram_c,
        stride_dgram_f,
        stride_scalar_c,
        stride_scalar_f,
        COMPUTE_DGRAM: tl.constexpr,
        COMPUTE_SCALAR: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_OUTPUTS: tl.constexpr,
    ):
        output = tl.program_id(0) * BLOCK_OUTPUTS + tl.arange(0, BLOCK_OUTPUTS)
        output_mask = output < _C_OUT * _NUM_FEATURES
        channel = output // _NUM_FEATURES
        feature = output - channel * _NUM_FEATURES
        accumulator = tl.zeros((BLOCK_OUTPUTS,), dtype=tl.float32)
        for split in range(SPLIT_K):
            accumulator += tl.load(
                partial_ptr
                + split * stride_partial_split
                + channel * stride_partial_c
                + feature * stride_partial_f,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)

        if COMPUTE_DGRAM:
            dgram_mask = output_mask & (feature < _NUM_BINS)
            tl.store(
                grad_dgram_ptr + channel * stride_dgram_c + feature * stride_dgram_f,
                accumulator,
                mask=dgram_mask,
            )
        if COMPUTE_SCALAR:
            scalar_mask = output_mask & (feature >= _NUM_BINS)
            tl.store(
                grad_scalar_ptr
                + channel * stride_scalar_c
                + (feature - _NUM_BINS) * stride_scalar_f,
                accumulator,
                mask=scalar_mask,
            )


def _coordinate_inputs_match(
    n_token: int,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
) -> bool:
    return (
        pseudo_beta_coords.shape == (1, n_token, 3)
        and frame_atom_coords.shape == (1, n_token, 3, 3)
        and pseudo_beta_mask.shape == (1, n_token)
        and backbone_frame_mask.shape == (1, n_token)
        and asym_id.shape == (1, n_token)
    )


def template_coordinate_projection(
    source: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    dgram_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add coordinate-derived projections to ``source`` and write ``out``.

    ``source`` and ``out`` may be the same tensor for guarded inference use.
    Training callers pass a fresh ``out`` through an autograd wrapper. This
    launcher intentionally supports only the production B=1, fp32/bf16, 39+5
    to 64 shape.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for coordinate template projection")
    if out is None:
        out = torch.empty_like(source)
    inputs = (
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
    )
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (source, out, *inputs)
    ):
        raise RuntimeError(
            "Raw template coordinate Triton launches are not autograd-aware; "
            "use the model dispatcher"
        )
    if not (
        source.is_cuda
        and out.is_cuda
        and source.device == out.device
        and source.dtype in (torch.float32, torch.bfloat16)
        and out.dtype == source.dtype
        and source.dim() == 4
        and source.shape[0] == 1
        and source.shape[-1] == 64
        and all(x.is_cuda and x.device == source.device for x in inputs)
        and dgram_weight.dtype == torch.float32
        and scalar_weight.dtype == torch.float32
        and dgram_weight.shape == (64, 39)
        and scalar_weight.shape == (64, 5)
    ):
        raise ValueError("Unsupported coordinate template projection configuration")

    n_token = source.shape[1]
    expected_shape = (1, n_token, n_token, 64)
    if source.shape != expected_shape or out.shape != expected_shape:
        raise ValueError(
            "Expected source and out [1,N,N,64], got "
            f"{tuple(source.shape)} and {tuple(out.shape)}"
        )
    if not _coordinate_inputs_match(n_token, *inputs[:5]):
        raise ValueError("Coordinate inputs do not match source token dimensions")
    if torch._debug_has_internal_overlap(out) != 0:
        raise ValueError("Template coordinate output must not overlap internally")
    if (
        source.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        and source is not out
    ):
        raise ValueError("source and out may only alias when they are the same tensor")

    pb = pseudo_beta_coords[0]
    frame = frame_atom_coords[0]
    pb_mask = pseudo_beta_mask[0]
    bb_mask = backbone_frame_mask[0]
    asym = asym_id[0]
    grid = (triton.cdiv(n_token * n_token, _BLOCK_PAIRS),)
    _template_coordinate_projection_kernel[grid](
        source,
        out,
        pb,
        frame,
        pb_mask,
        bb_mask,
        asym,
        dgram_weight,
        scalar_weight,
        n_token,
        source.stride(1),
        source.stride(2),
        source.stride(3),
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
    return out


def template_coordinate_projection_backward(
    grad_output: torch.Tensor,
    pseudo_beta_coords: torch.Tensor,
    frame_atom_coords: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    backbone_frame_mask: torch.Tensor,
    asym_id: torch.Tensor,
    *,
    compute_dgram: bool = True,
    compute_scalar: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute coordinate-projection weight gradients with deterministic split-K."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for coordinate template projection")
    if not compute_dgram and not compute_scalar:
        return None, None
    inputs = (
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
    )
    if not (
        grad_output.is_cuda
        and grad_output.dtype in (torch.float32, torch.bfloat16)
        and grad_output.dim() == 4
        and grad_output.shape[0] == 1
        and grad_output.shape[-1] == 64
        and all(x.is_cuda and x.device == grad_output.device for x in inputs)
    ):
        raise ValueError(
            "Unsupported coordinate template projection backward configuration"
        )

    n_token = grad_output.shape[1]
    if grad_output.shape != (1, n_token, n_token, 64):
        raise ValueError("Expected grad_output [1,N,N,64]")
    if not _coordinate_inputs_match(n_token, *inputs):
        raise ValueError("Coordinate inputs do not match grad_output token dimensions")

    grad = grad_output[0]
    pb = pseudo_beta_coords[0]
    frame = frame_atom_coords[0]
    pb_mask = pseudo_beta_mask[0]
    bb_mask = backbone_frame_mask[0]
    asym = asym_id[0]
    partial = torch.empty(
        (_BWD_SPLIT_K, 64, _BWD_FEATURES),
        dtype=torch.float32,
        device=grad_output.device,
    )
    grad_dgram = torch.empty((64, 39), dtype=torch.float32, device=grad_output.device)
    grad_scalar = torch.empty((64, 5), dtype=torch.float32, device=grad_output.device)

    partial_grid = (
        triton.cdiv(64, _BWD_BLOCK_CHANNELS),
        triton.cdiv(_BWD_FEATURES, _BWD_BLOCK_FEATURES),
        _BWD_SPLIT_K,
    )
    _template_coordinate_projection_bwd_partial_kernel[partial_grid](
        grad,
        pb,
        frame,
        pb_mask,
        bb_mask,
        asym,
        partial,
        n_token,
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        pb.stride(0),
        pb.stride(1),
        frame.stride(0),
        frame.stride(1),
        frame.stride(2),
        pb_mask.stride(0),
        bb_mask.stride(0),
        asym.stride(0),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        COMPUTE_DGRAM=compute_dgram,
        COMPUTE_SCALAR=compute_scalar,
        INPUT_PRECISION="tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee",
        BLOCK_PAIRS=_BWD_BLOCK_PAIRS,
        BLOCK_CHANNELS=_BWD_BLOCK_CHANNELS,
        BLOCK_FEATURES=_BWD_BLOCK_FEATURES,
        SPLIT_K=_BWD_SPLIT_K,
        num_warps=4,
        num_stages=1,
    )
    reduce_block = 256
    reduce_grid = (triton.cdiv(64 * _BWD_FEATURES, reduce_block),)
    _template_coordinate_projection_bwd_reduce_kernel[reduce_grid](
        partial,
        grad_dgram,
        grad_scalar,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        grad_dgram.stride(0),
        grad_dgram.stride(1),
        grad_scalar.stride(0),
        grad_scalar.stride(1),
        COMPUTE_DGRAM=compute_dgram,
        COMPUTE_SCALAR=compute_scalar,
        SPLIT_K=_BWD_SPLIT_K,
        BLOCK_OUTPUTS=reduce_block,
        num_warps=8,
        num_stages=1,
    )
    return (
        grad_dgram if compute_dgram else None,
        grad_scalar if compute_scalar else None,
    )


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
    """Inference-only in-place compatibility wrapper."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "In-place template coordinate projection requires disabled grad mode"
        )
    template_coordinate_projection(
        out,
        pseudo_beta_coords,
        frame_atom_coords,
        pseudo_beta_mask,
        backbone_frame_mask,
        asym_id,
        dgram_weight,
        scalar_weight,
        out=out,
    )
