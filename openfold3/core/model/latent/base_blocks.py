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

"""Base blocks for MSA-based and Pair transformer stacks. Includes MSABlock and PairBlock, where the
MSABlock is used to define the blocks used in the EvoformerStack, ExtraMSAStack, and MSAModule.
"""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from openfold3.core.model.layers.msa import MSARowAttentionWithPairBias
from openfold3.core.model.layers.outer_product_mean import OuterProductMean
from openfold3.core.model.layers.transition import ReLUTransition, SwiGLUTransition
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives import DropoutRowwise
from openfold3.core.utils.tensor_utils import add


class MSABlock(nn.Module, ABC):
    """Abstract class for MSA blocks. Used to define the blocks used in the EvoformerStack,
    ExtraMSAStack, and MSAModule.
    """

    @abstractmethod
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_type: str,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition function
            transition_n:
                Factor by which to multiply c_m to obtain the transition layer
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            opm_first:
                When True, Outer Product Mean is performed at the beginning of
                the Evoformer block instead of after the MSA Stack.
                Used in Multimer pipeline.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            inf:
                Large constant for masking
            eps:
                Small constant for numerical stability
        """
        super(MSABlock, self).__init__()

        self.opm_first = opm_first

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        if transition_type == "relu":
            self.msa_transition = ReLUTransition(
                c_in=c_m,
                n=transition_n,
            )
        elif transition_type == "swiglu":
            self.msa_transition = SwiGLUTransition(
                c_in=c_m,
                n=transition_n,
            )
        else:
            raise ValueError(f"Transition type {transition_type} is not available")

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_type=transition_type,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
        )

    def _compute_opm(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Outer Product Mean. Will be used in the forward pass of the MSABlock."""
        m, z = input_tensors

        if _offload_inference and inplace_safe:
            # m: GPU, z: CPU
            del m, z
            assert sys.getrefcount(input_tensors[1]) == 2
            input_tensors[1] = input_tensors[1].cpu()
            m, z = input_tensors

        opm = self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe)

        if _offload_inference and inplace_safe:
            # m: GPU, z: GPU
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        z = add(z, opm, inplace=inplace_safe)
        del opm

        return m, z

    @abstractmethod
    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class PairBlock(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_type: str,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
    ):
        """
        Args:
            c_z:
                Pair embedding channel dimension
            c_hidden_mul:
                Hidden dimension for triangular multiplication
            c_hidden_pair_att:
                Per-head hidden dimension for triangular attention
            no_heads_pair:
                Number of heads in the attention mechanism
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition function
            transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            pair_dropout:
                Dropout rate used throughout the stack
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            inf:
                Large constant used for masking
        """
        super(PairBlock, self).__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        if transition_type == "relu":
            self.pair_transition = ReLUTransition(
                c_in=c_z,
                n=transition_n,
            )
        elif transition_type == "swiglu":
            self.pair_transition = SwiGLUTransition(
                c_in=c_z,
                n=transition_n,
            )
        else:
            raise ValueError(f"Transition type {transition_type} is not available")

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def tri_mul_out_in(self, z: torch.Tensor, pair_mask: torch.Tensor, inplace_safe: bool) -> torch.Tensor:
        """Perform the outgoing and incoming triangular multiplicative updates."""
        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        return z

    def tri_att_start_end(
        self,
        z: torch.Tensor,
        _attn_chunk_size: Optional[int],
        pair_mask: torch.Tensor,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        inplace_safe: bool,
    ) -> torch.Tensor:
        """Perform the starting and ending triangular attention layers."""
        z = add(
            z,
            self.ps_dropout_row_layer(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            z = z.contiguous()

        z = add(
            z,
            self.ps_dropout_row_layer(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            z = z.contiguous()

        return z

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N, N, C_z] Pair embedding
            pair_mask:
                [*, N, N] Pair mask
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers
            _attn_chunk_size:
                Inference-time subbatch size for attention. If None, uses chunk.

        Returns:
            [*, N, N, C_z] Pair embedding update
        """

        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        z = self.tri_mul_out_in(z=z, pair_mask=pair_mask, inplace_safe=inplace_safe)

        z = self.tri_att_start_end(
            z=z,
            _attn_chunk_size=_attn_chunk_size,
            pair_mask=pair_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
        )

        z = add(
            z,
            self.pair_transition(
                z,
                mask=pair_trans_mask,
                chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        return z
