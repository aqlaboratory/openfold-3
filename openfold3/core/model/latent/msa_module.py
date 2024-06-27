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

"""MSA module block and stack. Note that this does not include the MSA sampling, which is handled
in the MSAModuleEmbedder.
"""

from typing import Optional

import torch

from openfold3.core.model.latent.base_stacks import MSAStack
from openfold3.core.model.latent.evoformer import EvoformerBlock
from openfold3.core.model.layers.msa import MSAPairWeightedAveraging


class MSAModuleBlock(EvoformerBlock):
    """Implements block of AF3 Algorithm 8."""
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
                the MSAModule block instead of after the MSA Stack.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            inf:
                Large constant for masking
            eps:
                Small constant for numerical stability
        """
        super(MSAModuleBlock, self).__init__(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_type=transition_type,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            no_column_attention=True,
            opm_first=opm_first,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps
        )

        # Column attention is disabled and MSAPairWeightedAveraging replace MSARowAttentionWithPairBias
        self.msa_att_row = MSAPairWeightedAveraging(
            c_in=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )


class MSAModuleStack(MSAStack):
    """Implements AF3 Algorithm 8 lines 5-15. The MSA sampling and initial embedding is
    handled in MSAModuleEmbedder prior to calling this stack.
    """
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
        no_blocks: int,
        transition_type: str,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        blocks_per_ckpt: Optional[int],
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        **kwargs,
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
            no_blocks:
                Number of MSAModule blocks in the stack
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
                the MSAModule block instead of after the MSA Stack.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of MSAModule blocks in each activation checkpoint
            inf:
                Large constant for masking
            eps:
                Small constant for numerical stability
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(MSAModuleStack, self).__init__(
            blocks_per_ckpt=blocks_per_ckpt,
            clear_cache_between_blocks=clear_cache_between_blocks,
            tune_chunk_size=tune_chunk_size
        )

        for _ in range(no_blocks):
            block = MSAModuleBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_type=transition_type,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

    def _wrap_up(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return only the pair embedding.

        Returns:
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """
        return z
