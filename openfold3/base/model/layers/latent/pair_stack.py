from typing import Optional

import torch
import torch.nn as nn

from openfold3.base.model.layers.modules.triangular_attention import TriangleAttention
from openfold3.base.model.layers.modules.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
    FusedTriangleMultiplicationIncoming
)
from openfold3.base.model.layers.modules.transition import ReLUTransition, SwiGLUTransition
from openfold3.base.model.primitives import DropoutRowwise
from openfold3.base.utils.tensor_utils import add


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        transition_type: str = 'relu'
    ):
        super(PairStack, self).__init__()

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

        if transition_type == 'relu':
            self.pair_transition = ReLUTransition(
                c_in=c_z,
                n=transition_n,
            )
        elif transition_type == 'swiglu':
            self.pair_transition = SwiGLUTransition(
                c_in=c_z,
                n=transition_n,
            )
        else:
            raise ValueError(f'Transition type {transition_type} is not available')

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
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
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
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
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
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
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )

        return z


