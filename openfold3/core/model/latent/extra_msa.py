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

"""Extra MSA block and stack."""

import sys
from collections.abc import Sequence
from typing import Optional

import torch
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.base_blocks import MSABlock
from openfold3.core.model.latent.base_stacks import MSAStack
from openfold3.core.model.layers.msa import MSAColumnGlobalAttention
from openfold3.core.utils.checkpointing import get_checkpoint_fn
from openfold3.core.utils.tensor_utils import add


class ExtraMSABlock(MSABlock):
    """
    Almost identical to the standard EvoformerBlock, except in that the
    ExtraMSABlock uses GlobalAttention for MSA column attention and
    requires more fine-grained control over checkpointing. Separated from
    its twin to preserve the TorchScript-ability of the latter.
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
        transition_type: str,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        ckpt: bool,
        linear_init_params: ConfigDict = lin_init.extra_msa_block_init,
    ):
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

        self.ckpt = ckpt

        self.msa_att_col = MSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps,
            linear_init_params=linear_init_params.msa_col_att,
        )

    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        transition_ckpt_chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        m = add(
            m,
            self.msa_dropout_layer(
                self.msa_att_row(
                    m.clone() if torch.is_grad_enabled() else m,
                    z=z.clone() if torch.is_grad_enabled() else z,
                    mask=msa_mask,
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_memory_efficient_kernel=not (
                        use_lma or use_deepspeed_evo_attention
                    ),
                    _checkpoint_chunks=self.ckpt if torch.is_grad_enabled() else False,
                )
            ),
            inplace=inplace_safe,
        )

        if not inplace_safe:
            input_tensors = [m, z]

        del m, z

        def fn(input_tensors):
            m, z = input_tensors

            if _offload_inference and inplace_safe:
                # m: GPU, z: CPU
                del m, z
                assert sys.getrefcount(input_tensors[1]) == 2
                input_tensors[1] = input_tensors[1].cpu()
                torch.cuda.empty_cache()
                m, z = input_tensors

            m = add(
                m,
                self.msa_att_col(
                    m,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                    use_lma=use_lma,
                ),
                inplace=inplace_safe,
            )

            m = add(
                m,
                self.msa_transition(
                    m,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                    ckpt_chunk_size=transition_ckpt_chunk_size,
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
                input_tensors[1],
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
                _attn_chunk_size=_attn_chunk_size,
            )

            m = input_tensors[0]
            if _offload_inference and inplace_safe:
                # m: GPU, z: GPU
                device = z.device
                del m
                assert sys.getrefcount(input_tensors[0]) == 2
                input_tensors[0] = input_tensors[0].to(device)
                m, _ = input_tensors

            return m, z

        if torch.is_grad_enabled() and self.ckpt:
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, input_tensors)
        else:
            m, z = fn(input_tensors)

        return m, z


class ExtraMSAStack(MSAStack):
    """Implements AF2 Algorithm 18."""

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
        inf: float,
        eps: float,
        ckpt: bool,
        linear_init_params: ConfigDict = lin_init.extra_msa_block_init,
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
                Number of ExtraMSA blocks in the stack
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            transition_n:
                Factor by which to multiply c_m to obtain the transition layer hidden
                dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            no_column_attention:
                When True, doesn't use column attention. Required for running sequence
                embedding mode
            opm_first:
                When True, Outer Product Mean is performed at the beginning of the
                ExtraMSA block instead of after the MSA Stack. Used in Multimer
                pipeline.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in the Pair
                Stack. Used in Multimer pipeline.
            inf:
                Large constant for masking
            eps:
                Small constant for numerical stability
            ckpt:
                Whether to checkpoint blocks
            linear_init_params:
                Parameters for linear layer initialization
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the stack.
                Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        # Previous implementation seems to call checkpoint function once for all blocks
        blocks_per_ckpt = None if not ckpt else no_blocks
        super().__init__(
            blocks_per_ckpt=blocks_per_ckpt,
            use_reentrant=use_reentrant,
            clear_cache_between_blocks=clear_cache_between_blocks,
            tune_chunk_size=tune_chunk_size,
        )

        self.ckpt = ckpt

        for _ in range(no_blocks):
            block = ExtraMSABlock(
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
                ckpt=False,
                linear_init_params=linear_init_params,
            )
            self.blocks.append(block)

    def _wrap_up(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return only the pair embedding.

        Returns:
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """
        return z
