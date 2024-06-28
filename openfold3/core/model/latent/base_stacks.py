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

"""Base MSAStack that is used to define the following: EvoformerStack, ExtraMSAStack, and MSAModule."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from openfold3.core.utils.checkpointing import checkpoint_blocks
from openfold3.core.utils.chunk_utils import ChunkSizeTuner


# TODO: Rename to CheckpointStack and generalize any kind of block (i.e. remove references to m/z)
class MSAStack(nn.Module, ABC):
    """Abstract class for MSA stacks."""

    @abstractmethod
    def __init__(
        self,
        blocks_per_ckpt: Optional[int],
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            blocks_per_ckpt:
                Number of blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(MSAStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        # Must be composed in subclasses
        self.blocks = nn.ModuleList()

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        """
        Partially initialize the blocks. Optionally add cache clearing between
        blocks and chunk size tuning. Arguments are the same as forward function.

        Returns:
            Partially initialized blocks.
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # Tensors cloned to avoid getting written to in-place
                # A corollary is that chunk size tuning should be disabled for
                # large N, when z gets really big
                args=(m.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )

            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                ) for b in blocks
            ]

        return blocks

    def _wrap_up(self, m: torch.Tensor, z: torch.Tensor):
        """Function called at the end of the forward pass to wrap up the outputs and return
        the appropriate tensors depending on the stack type.

        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] Pair embedding
        """
        return m, z

    def forward_offload(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        _mask_trans: bool = True,
    ):
        assert not (self.training or torch.is_grad_enabled())

        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            m, z = b(
                None,
                None,
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z

        m, z = input_tensors

        return self._wrap_up(m, z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            use_flash:
                Whether to use FlashAttention where possible. Mutually
                exclusive with use_lma and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
        """
        blocks = self._prep_blocks(
            m=m,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        return self._wrap_up(m, z)
