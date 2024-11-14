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

"""MSA module block and stack.

Note that this does not include the MSA sampling, which is handled in the
MSAModuleEmbedder.
"""

import sys
from typing import Optional, Sequence, Tuple

import torch
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.base_blocks import MSABlock
from openfold3.core.model.latent.base_stacks import MSAStack
from openfold3.core.model.layers.msa import MSAPairWeightedAveraging
from openfold3.core.utils.tensor_utils import add


class MSAModuleBlock(MSABlock):
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
        transition_ckpt_chunk_size: Optional[int] = None,
        linear_init_params: ConfigDict = lin_init.msa_module_init,
        last_block: bool = False,
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
                String 'relu' or 'swiglu' to determine activation for the transition
                function
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
            transition_ckpt_chunk_size:
                Chunk size for activation checkpointing in the transition layer
                (currently SwiGLU transition only)
            linear_init_params:
                Parameters for linear layer initialization
            last_block:
                Whether this is the last block and the msa embedding updates should
                be skipped
        """
        super().__init__(
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
            linear_init_params=linear_init_params,
        )

        self.transition_ckpt_chunk_size = transition_ckpt_chunk_size
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            # Column attention is disabled and MSAPairWeightedAveraging replace
            # MSARowAttentionWithPairBias
            self.msa_att_row = MSAPairWeightedAveraging(
                c_in=c_m,
                c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
                linear_init_params=linear_init_params.msa_pair_avg,
            )
        else:
            self.msa_att_row = None
            self.msa_dropout_layer = None
            self.msa_transition = None

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
        msa_trans_mask = msa_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
            )

        if not self.skip_msa_update:
            m = add(
                m,
                self.msa_dropout_layer(
                    self.msa_att_row(
                        m,
                        z=z,
                        mask=pair_mask,
                    )
                ),
                inplace=inplace_safe,
            )

            if _offload_inference and inplace_safe:
                # m: GPU, z: CPU
                del m, z
                assert sys.getrefcount(input_tensors[1]) == 2
                input_tensors[1] = input_tensors[1].cpu()
                torch.cuda.empty_cache()
                m, z = input_tensors

            m = add(
                m,
                self.msa_transition(
                    m,
                    mask=msa_trans_mask,
                    chunk_size=chunk_size,
                    ckpt_chunk_size=self.transition_ckpt_chunk_size,
                ),
                inplace=inplace_safe,
            )

            if not self.opm_first:
                if not inplace_safe:
                    input_tensors = [m, z]

                del m, z

                m, z = self._compute_opm(
                    input_tensors=input_tensors,
                    msa_mask=msa_mask,
                    chunk_size=chunk_size,
                    inplace_safe=inplace_safe,
                    _offload_inference=_offload_inference,
                )

        if _offload_inference and inplace_safe:
            # m: CPU, z: GPU
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            device = input_tensors[0].device
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        if not inplace_safe:
            input_tensors = [m, z]

        del m, z

        z = self.pair_stack(
            z=input_tensors[1],
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        if _offload_inference and inplace_safe:
            # m: GPU, z: GPU
            device = z.device
            assert sys.getrefcount(input_tensors[0]) == 2
            input_tensors[0] = input_tensors[0].to(device)
            m, _ = input_tensors
        else:
            m = input_tensors[0]

        return m, z


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
        transition_ckpt_chunk_size: Optional[int] = None,
        linear_init_params: ConfigDict = lin_init.msa_module_init,
        use_reentrant: Optional[bool] = None,
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
                String 'relu' or 'swiglu' to determine activation for the transition
                function
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
            transition_ckpt_chunk_size:
                Chunk size for activation checkpointing in the transition layer
                (currently SwiGLU transition only)
            linear_init_params:
                Parameters for linear layer initialization
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__(
            blocks_per_ckpt=blocks_per_ckpt,
            use_reentrant=use_reentrant,
            clear_cache_between_blocks=clear_cache_between_blocks,
            tune_chunk_size=tune_chunk_size,
        )

        for i in range(no_blocks):
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
                transition_ckpt_chunk_size=transition_ckpt_chunk_size,
                linear_init_params=linear_init_params,
                last_block=i == no_blocks - 1,
            )
            self.blocks.append(block)

    def _wrap_up(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return only the pair embedding.

        Returns:
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """
        return z
