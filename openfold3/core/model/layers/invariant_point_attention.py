import importlib
import math
import sys
from _operator import mul
from collections.abc import Sequence
from functools import reduce
from typing import Optional, Tuple, Union

import torch
from ml_collections import ConfigDict
from torch import nn as nn

from openfold3.core.config import default_linear_init_config as lin_init
from openfold3.core.model.primitives import Linear, ipa_point_weights_init_
from openfold3.core.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold3.core.utils.geometry.vector import Vec3Array, square_euclidean_distance
from openfold3.core.utils.precision_utils import is_fp16_enabled
from openfold3.core.utils.rigid_utils import Rigid
from openfold3.core.utils.tensor_utils import flatten_final_dims, permute_final_dims

# To avoid errors if memory-efficient attention kernel is not installed
attn_core_is_installed = importlib.util.find_spec("attn_core_inplace_cuda") is not None
if attn_core_is_installed:
    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class PointProjection(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        precision: Optional[torch.dtype] = None,
        return_local_points: bool = False,
        linear_init_params: ConfigDict = lin_init.point_proj_init,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points

        # Multimer requires this to be run with fp32 precision during training
        self.linear = Linear(
            c_hidden,
            no_heads * 3 * num_points,
            precision=precision,
            **linear_init_params.linear,
        )

    def forward(
        self,
        activations: torch.Tensor,
        rigids: Union[Rigid, Rigid3Array],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

        if isinstance(rigids, Rigid3Array):
            points_local = points_local.view(
                points_local.shape[:-1] + (self.no_heads, -1)
            )

        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)

        points_local = torch.stack(points_local, dim=-1).view(out_shape)

        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements AF2 Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        linear_init_params: ConfigDict = lin_init.monomer_ipa_init,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
            linear_init_params:
                Initialization parameters for linear layers
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, **linear_init_params.linear_q)

        self.linear_q_points = PointProjection(
            c_hidden=self.c_s,
            num_points=self.no_qk_points,
            no_heads=self.no_heads,
            linear_init_params=linear_init_params.linear_q_points,
        )

        self.linear_kv = Linear(self.c_s, 2 * hc, **linear_init_params.linear_kv)
        self.linear_kv_points = PointProjection(
            c_hidden=self.c_s,
            num_points=self.no_qk_points + self.no_v_points,
            no_heads=self.no_heads,
            linear_init_params=linear_init_params.linear_kv_points,
        )

        self.linear_b = Linear(self.c_z, self.no_heads, **linear_init_params.linear_b)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(
            concat_out_dim, self.c_s, **linear_init_params.linear_out
        )

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, P_qk]
        q_pts = self.linear_q_points(s, r)

        # [*, N_res, H * 2 * C_hidden]
        kv = self.linear_kv(s)

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        kv_pts = self.linear_kv_points(s, r)

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            assert sys.getrefcount(z[0]) == 2
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        # Einsum notation: [H:h, N_res:(n,m), C_hidden:c]
        # n,m are used because using n,n would be ambiguous einstein notation
        if is_fp16_enabled():
            with torch.amp.autocast("cuda", enabled=False):
                a = torch.einsum("...nhc,...mhc->...hnm", q.float(), k.float())
        else:
            a = torch.einsum("...nhc,...mhc->...hnm", q, k)

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        if inplace_safe:
            pt_att *= pt_att
        else:
            pt_att = pt_att**2

        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        if inplace_safe:
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if inplace_safe and attn_core_is_installed:
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        ################
        # Compute output
        ################

        # This einsum is equivalent to:
        # Transpose v : [*, N_res, H, C_hidden] -> [*, H, N_res, C_hidden]
        # Matmul a, v: [*, H, N_res, N_res] x [*, H, N_res, C_hidden]
        #               -> [*, H, N_res, C_hidden]
        # Transpose o: [*, H, N_res, C_hidden] -> [*, N_res, H, C_hidden]
        # Einsum notation: [H:h, N_res:(n,m), C_hidden:c]
        # n,m are used because using n,n in output would be ambiguous einstein notation
        o = torch.einsum("...hnm,...mhc->...nhc", a, v.to(dtype=a.dtype))

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # IMPORTANT: This has been changed from the original version where there was
        # a very particular indexing to ensure fp32; if precision problems occur,
        # this is a place to look into.
        with torch.amp.autocast("cuda", enabled=False):
            o_pt = torch.einsum("...hnm, ...mhpt->...nhpt", a, v_pts.to(dtype=a.dtype))

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt = torch.unbind(o_pt, dim=-1)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        # This einsum is equivalent to:
        # Transpose a : [*, H, N_res, N_res] -> [*, N_res, H, N_res]
        # Matmul a, z: [*, N_res, H, N_res] x [*, N_res, N_res, C_z]
        #               -> [*, N_res, H, C_z]
        o_pair = torch.einsum("...hnm,...nmc->...nhc", a, z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *o_pt, o_pt_norm, o_pair), dim=-1).to(dtype=z[0].dtype)
        )

        return s


class InvariantPointAttentionMultimer(nn.Module):
    """
    Implements AF2-Multimer version of AF2 Algorithm 22.

    Note: This module follows the refactoring done in IPA for multimer.
    The functionality should be the same as the original InvariantPointAttention,
    save for a few linear layer changes.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        linear_init_params: ConfigDict = lin_init.multimer_ipa_init,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
            linear_init_params:
                Initialization parameters for linear layers
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, **linear_init_params.linear_q)

        self.linear_q_points = PointProjection(
            c_hidden=self.c_s,
            num_points=self.no_qk_points,
            no_heads=self.no_heads,
            precision=torch.float32,
            linear_init_params=linear_init_params.linear_q_points,
        )

        self.linear_k = Linear(self.c_s, hc, **linear_init_params.linear_k)
        self.linear_v = Linear(self.c_s, hc, **linear_init_params.linear_v)
        self.linear_k_points = PointProjection(
            c_hidden=self.c_s,
            num_points=self.no_qk_points,
            no_heads=self.no_heads,
            precision=torch.float32,
            linear_init_params=linear_init_params.linear_k_points,
        )

        self.linear_v_points = PointProjection(
            c_hidden=self.c_s,
            num_points=self.no_v_points,
            no_heads=self.no_heads,
            precision=torch.float32,
            linear_init_params=linear_init_params.linear_v_points,
        )

        self.linear_b = Linear(self.c_z, self.no_heads, **linear_init_params.linear_b)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(
            concat_out_dim, self.c_s, **linear_init_params.linear_out
        )

        self.softmax = nn.Softmax(dim=-2)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        a = 0.0

        point_variance = max(self.no_qk_points, 1) * 9.0 / 2
        point_weights = math.sqrt(1.0 / point_variance)

        # Apply softplus to head weights
        head_weights = torch.logaddexp(
            self.head_weights, torch.zeros_like(self.head_weights)
        )
        point_weights = point_weights * head_weights

        #######################################
        # Generate scalar and point activations
        #######################################

        # [*, N_res, H, P_qk]
        q_pts = Vec3Array.from_array(self.linear_q_points(s, r))

        # [*, N_res, H, P_qk, 3]
        k_pts = Vec3Array.from_array(self.linear_k_points(s, r))

        pt_att = square_euclidean_distance(
            q_pts.unsqueeze(-3), k_pts.unsqueeze(-4), epsilon=0.0
        )
        pt_att = torch.sum(pt_att * point_weights[..., None], dim=-1) * (-0.5)
        pt_att = pt_att.to(dtype=s.dtype)
        a = a + pt_att

        scalar_variance = max(self.c_hidden, 1) * 1.0
        scalar_weights = math.sqrt(1.0 / scalar_variance)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        k = self.linear_k(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))

        q = q * scalar_weights
        a = a + torch.einsum("...qhc,...khc->...qkh", q, k)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            assert sys.getrefcount(z[0]) == 2
            z[0] = z[0].cpu()

        a = a + b

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-1)
        a = a * math.sqrt(1.0 / 3)  # Normalize by number of logit terms (3)
        a = self.softmax(a)

        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        o = torch.einsum("...qkh, ...khc->...qhc", a, v)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v, 3]
        v_pts = Vec3Array.from_array(self.linear_v_points(s, r))

        # [*, N_res, H, P_v]
        o_pt = v_pts[..., None, :, :, :] * a.unsqueeze(-1)
        o_pt = o_pt.sum(dim=-3)
        # o_pt = Vec3Array(
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].x, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].y, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].z, dim=-3),
        # )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

        # [*, N_res, H, P_v]
        o_pt = r[..., None].apply_inverse_to_point(o_pt)
        o_pt_flat = [o_pt.x, o_pt.y, o_pt.z]
        o_pt_flat = [x.to(dtype=a.dtype) for x in o_pt_flat]

        # [*, N_res, H * P_v]
        o_pt_norm = o_pt.norm(epsilon=1e-8)

        if _offload_inference:
            z[0] = z[0].to(o_pt.x.device)

        o_pair = torch.einsum("...ijh, ...ijc->...ihc", a, z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *o_pt_flat, o_pt_norm, o_pair), dim=-1).to(dtype=z[0].dtype)
        )

        return s
