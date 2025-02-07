# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""AF2 Structure Module."""

import sys
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.angle_resnet import AngleResnet
from openfold3.core.model.layers.backbone_update import BackboneUpdate, QuatRigidUpdate
from openfold3.core.model.layers.invariant_point_attention import (
    InvariantPointAttention,
    InvariantPointAttentionMultimer,
)
from openfold3.core.model.layers.transition import StructureModuleTransition
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold3.core.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold3.core.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold3.core.utils.rigid_utils import Rigid, Rotation
from openfold3.core.utils.tensor_utils import (
    dict_multimap,
)


class BaseStructureModule(nn.Module, ABC):
    """Abstract class for StructureModule."""

    @abstractmethod
    def __init__(
        self,
        c_s,
        c_z,
        c_resnet,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        linear_init_params,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            linear_init_params:
                Parameters used for linear layer initialization
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z

        self.c_resnet = c_resnet
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s, **linear_init_params.linear_in)

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
            linear_init_params=linear_init_params.transition,
        )

        self.angle_resnet = AngleResnet(
            c_in=self.c_s,
            c_hidden=self.c_resnet,
            no_blocks=self.no_resnet_blocks,
            no_angles=self.no_angles,
            epsilon=self.epsilon,
            linear_init_params=linear_init_params.angle_resnet,
        )

    @abstractmethod
    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        pass

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self,
        r,
        f,  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.dtype, r.device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


class StructureModuleMonomer(BaseStructureModule):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        linear_init_params=lin_init.monomer_structure_module_init,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            linear_init_params:
                Parameters used for linear layer initialization
        """
        super().__init__(
            c_s=c_s,
            c_z=c_z,
            c_resnet=c_resnet,
            dropout_rate=dropout_rate,
            no_blocks=no_blocks,
            no_transition_layers=no_transition_layers,
            no_resnet_blocks=no_resnet_blocks,
            no_angles=no_angles,
            trans_scale_factor=trans_scale_factor,
            epsilon=epsilon,
            inf=inf,
            linear_init_params=linear_init_params,
        )

        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            linear_init_params=linear_init_params.ipa,
        )

        self.bb_update = BackboneUpdate(
            self.c_s, linear_init_params=linear_init_params.bb_update
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
        )
        outputs = []
        for _ in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs


class StructureModuleMultimer(BaseStructureModule):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        linear_init_params=lin_init.multimer_structure_module_init,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            linear_init_params:
                Parameters used for linear layer initialization
        """
        super().__init__(
            c_s=c_s,
            c_z=c_z,
            c_resnet=c_resnet,
            dropout_rate=dropout_rate,
            no_blocks=no_blocks,
            no_transition_layers=no_transition_layers,
            no_resnet_blocks=no_resnet_blocks,
            no_angles=no_angles,
            trans_scale_factor=trans_scale_factor,
            epsilon=epsilon,
            inf=inf,
            linear_init_params=linear_init_params,
        )

        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points

        self.ipa = InvariantPointAttentionMultimer(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            linear_init_params=linear_init_params.ipa,
        )

        self.bb_update = QuatRigidUpdate(
            self.c_s,
            full_quat=False,
            linear_init_params=linear_init_params.bb_update,
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for _ in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.bb_update(s)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids.scale_translation(self.trans_scale_factor),
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs
