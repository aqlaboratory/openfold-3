import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial

from .pair_stack import PairStack
from openfold3.base.model.layers.modules.attention_pair_bias import AttentionPairBias
from openfold3.base.model.layers.modules.transition import SwiGLUTransition
from openfold3.base.utils.checkpointing import checkpoint_blocks
from openfold3.base.utils.chunk_utils import ChunkSizeTuner
from openfold3.base.utils.tensor_utils import add


class PairFormerBlock(nn.Module):
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        super(PairFormerBlock, self).__init__()

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps,
            transition_type='swiglu'
        )

        self.attn_pair_bias = AttentionPairBias(c_q=c_s,
                                                c_k=c_s,
                                                c_v=c_s,
                                                c_z=c_z,
                                                c_hidden=c_hidden_pair_bias,
                                                no_heads=no_heads_pair_bias,
                                                use_ada_layer_norm=False,
                                                gating=True,
                                                inf=inf)

        self.single_transition = SwiGLUTransition(
            c_in=c_s,
            n=transition_n,
        )

    def forward(self,
                s: Optional[torch.Tensor],
                z: Optional[torch.Tensor],
                single_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                chunk_size: Optional[int] = None,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _mask_trans: bool = True,
                _attn_chunk_size: Optional[int] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(
            z=z,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size
        )

        s = add(s,
                self.attention_pair_bias(a=s, z=z, s=None, beta=None, mask=single_mask,
                                         use_memory_efficient_kernel=False,
                                         use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                         use_lma=use_lma),
                inplace=inplace_safe,
                )

        s = add(
            s,
            self.single_transition(
                s, mask=single_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        return s, z


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
                Channel dimension of the output "single" embedding
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
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the ReLUTransition
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
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
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