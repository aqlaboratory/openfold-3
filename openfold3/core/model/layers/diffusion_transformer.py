from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .attention_pair_bias import AttentionPairBias
from .transition import ConditionedTransitionBlock


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        inf: float = 1e9
    ):
        """

        Args:
            c_s:
            c_z:
            c_hidden:
            no_heads:
            n_transition:
            inf:
        """
        super(DiffusionTransformerBlock, self).__init__()

        self.attention_pair_bias = AttentionPairBias(c_q=c_s,
                                                     c_k=c_s,
                                                     c_v=c_s,
                                                     c_z=c_z,
                                                     c_hidden=c_hidden,
                                                     no_heads=no_heads,
                                                     use_ada_layer_norm=True,
                                                     gating=True,
                                                     inf=inf)

        self.conditioned_transition = ConditionedTransitionBlock(c_in=c_s, n=n_transition)

    def forward(self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        beta: Optional[torch.Tensor],
        mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """

        Args:
            a:
            s:
            z:
            beta:
            mask:
            use_memory_efficient_kernel:
            use_deepspeed_evo_attention:
            use_lma:
            _mask_trans:

        Returns:

        """
        b = self.attention_pair_bias(a=a, z=z, s=s, beta=beta, mask=mask,
                                     use_memory_efficient_kernel=use_memory_efficient_kernel,
                                     use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                     use_lma=use_lma)

        trans_mask = mask if _mask_trans else None
        a = b + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """
    Implements AF3 Algorithm 23.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        inf: float,
    ):
        """

        Args:
            c_s:
            c_z:
            c_hidden:
            no_heads:
            no_blocks:
            n_transition:
            inf:
        """
        super(DiffusionTransformer, self).__init__()

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                n_transition=n_transition,
                inf=inf
            )
            for _ in range(no_blocks)])

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                z: Optional[torch.Tensor],
                beta: Optional[torch.Tensor],
                mask: torch.Tensor,
                use_memory_efficient_kernel: bool = False,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                _mask_trans: bool = True
                ) -> torch.Tensor:
        """

        Args:
            a:
            s:
            z:
            beta:
            mask:
            use_memory_efficient_kernel:
            use_deepspeed_evo_attention:
            use_lma:
            _mask_trans:

        Returns:

        """
        # Do we need all the fancy checkpoint blocks from evoformer?
        blocks = [
            partial(
                b,
                s=s,
                z=z,
                beta=beta,
                mask=mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        for b in blocks:
            a = b(a)

        return a
