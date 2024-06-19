from typing import Optional, Tuple

import torch
import torch.nn as nn

from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.model.layers.triangular_attention import (
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
    FusedTriangleMultiplicationIncoming
)
from openfold3.core.model.layers.transition import ReLUTransition, SwiGLUTransition
from openfold3.core.model.primitives import DropoutRowwise, DropoutColumnwise
from openfold3.core.utils.tensor_utils import add


class PairStackBlock(nn.Module):
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
        super(PairStackBlock, self).__init__()

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


class TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        tri_mul_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        **kwargs,
    ):
        super(TemplatePairStackBlock, self).__init__()

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf
        self.tri_mul_first = tri_mul_first

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                self.c_t,
                self.c_hidden_tri_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                self.c_t,
                self.c_hidden_tri_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                self.c_t,
                self.c_hidden_tri_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                self.c_t,
                self.c_hidden_tri_mul,
            )

        self.pair_transition = ReLUTransition(
            c_in=self.c_t,
            n=self.pair_transition_n,
        )

    def tri_att_start_end(self,
                          single: torch.Tensor,
                          _attn_chunk_size: Optional[int],
                          single_mask: torch.Tensor,
                          use_deepspeed_evo_attention: bool,
                          use_lma: bool,
                          inplace_safe: bool):
        single = add(single,
                     self.dropout_row(
                         self.tri_att_start(
                             single,
                             chunk_size=_attn_chunk_size,
                             mask=single_mask,
                             use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                             use_lma=use_lma,
                             inplace_safe=inplace_safe,
                         )
                     ),
                     inplace_safe,
                     )

        single = add(single,
                     self.dropout_col(
                         self.tri_att_end(
                             single,
                             chunk_size=_attn_chunk_size,
                             mask=single_mask,
                             use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                             use_lma=use_lma,
                             inplace_safe=inplace_safe,
                         )
                     ),
                     inplace_safe,
                     )

        return single

    def tri_mul_out_in(self,
                       single: torch.Tensor,
                       single_mask: torch.Tensor,
                       inplace_safe: bool):
        tmu_update = self.tri_mul_out(
            single,
            mask=single_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            single = single + self.dropout_row(tmu_update)
        else:
            single = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            single,
            mask=single_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            single = single + self.dropout_row(tmu_update)
        else:
            single = tmu_update

        del tmu_update

        return single

    def forward(self,
                z: torch.Tensor,
                mask: torch.Tensor,
                chunk_size: Optional[int] = None,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _mask_trans: bool = True,
                _attn_chunk_size: Optional[int] = None,
                ):
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        single_templates = [
            t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)
        ]
        single_templates_masks = [
            m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
        ]

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]

            if self.tri_mul_first:
                single = self.tri_att_start_end(single=self.tri_mul_out_in(single=single,
                                                                           single_mask=single_mask,
                                                                           inplace_safe=inplace_safe),
                                                _attn_chunk_size=_attn_chunk_size,
                                                single_mask=single_mask,
                                                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                                use_lma=use_lma,
                                                inplace_safe=inplace_safe)
            else:
                single = self.tri_mul_out_in(
                    single=self.tri_att_start_end(single=single,
                                                  _attn_chunk_size=_attn_chunk_size,
                                                  single_mask=single_mask,
                                                  use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                                  use_lma=use_lma,
                                                  inplace_safe=inplace_safe),
                    single_mask=single_mask,
                    inplace_safe=inplace_safe)

            single = add(single,
                         self.pair_transition(
                             single,
                             mask=single_mask if _mask_trans else None,
                             chunk_size=chunk_size,
                         ),
                         inplace_safe,
                         )

            if not inplace_safe:
                single_templates[i] = single

        if not inplace_safe:
            z = torch.cat(single_templates, dim=-4)

        return z


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

        self.pair_stack = PairStackBlock(
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
                self.attn_pair_bias(a=s, z=z, s=None, beta=None, mask=single_mask,
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
