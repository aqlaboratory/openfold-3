import torch
from typing import Tuple, Optional
from functools import partial

from torch import nn as nn

from .pair_blocks import TemplatePairStackBlock, PairFormerBlock
from openfold3.core.model.primitives import LayerNorm
from openfold3.core.utils.checkpointing import checkpoint_blocks
from openfold3.core.utils.chunk_utils import ChunkSizeTuner


class TemplatePairStack(nn.Module):
    """
    Implements AF2 Algorithm 16.
    """

    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        dropout_rate,
        tri_mul_first,
        fuse_projection_weights,
        blocks_per_ckpt,
        tune_chunk_size: bool = False,
        inf=1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(TemplatePairStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                tri_mul_first=tri_mul_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert (not self.training)
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                args=(t.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b,
                        chunk_size=tuned_chunk_size,
                        _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                        ) for b in blocks
            ]

        t, = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        t = self.layer_norm(t)

        return t


class PairFormerStack(nn.Module):
    """
    Implements AF3 Algorithm 17.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float,
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
            c_s:
            c_z:
            c_hidden_pair_bias:
            no_heads_pair_bias:
            c_hidden_mul:
            c_hidden_pair_att:
            no_heads_pair:
            no_blocks:
            transition_n:
            pair_dropout:
            fuse_projection_weights:
            blocks_per_ckpt:
            inf:
            eps:
            clear_cache_between_blocks:
            tune_chunk_size:
            **kwargs:
        """
        super(PairFormerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = PairFormerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if (tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(self,
                     s: torch.Tensor,
                     z: torch.Tensor,
                     chunk_size: int,
                     use_deepspeed_evo_attention: bool,
                     use_lma: bool,
                     single_mask: Optional[torch.Tensor],
                     pair_mask: Optional[torch.Tensor],
                     inplace_safe: bool,
                     _mask_trans: bool,
                     ):
        blocks = [
            partial(
                b,
                single_mask=single_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if (self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if (chunk_size is not None and self.chunk_size_tuner is not None):
            assert (not self.training)
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(s.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b,
                        chunk_size=tuned_chunk_size,
                        # A temporary measure to address torch's occasional
                        # inability to allocate large tensors
                        _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                        ) for b in blocks
            ]

        return blocks

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                single_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                chunk_size: int,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _mask_trans: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            single_mask:
                [*, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
        Returns:
            s:
                [*, N_res, C_m] Single embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
        """
        blocks = self._prep_blocks(
            s=s,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            single_mask=single_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if (not torch.is_grad_enabled()):
            blocks_per_ckpt = None

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        return s, z

