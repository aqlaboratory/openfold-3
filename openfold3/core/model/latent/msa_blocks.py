import sys
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple

import torch
import torch.nn as nn

from .pair_blocks import PairStackBlock
from openfold3.core.model.layers.msa import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    MSAColumnGlobalAttention,
    MSAPairWeightedAveraging
)
from openfold3.core.model.layers.outer_product_mean import OuterProductMean
from openfold3.core.model.layers.transition import ReLUTransition, SwiGLUTransition
from openfold3.core.model.primitives import DropoutRowwise
from openfold3.core.utils.checkpointing import get_checkpoint_fn
from openfold3.core.utils.tensor_utils import add


class MSABlock(nn.Module, ABC):
    @abstractmethod
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_type:str,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
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

        if transition_type == 'relu':
            self.msa_transition = ReLUTransition(
                c_in=c_m,
                n=transition_n,
            )
        elif transition_type == 'swiglu':
            self.msa_transition = SwiGLUTransition(
                c_in=c_m,
                n=transition_n,
            )
        else:
            raise ValueError(f'Transition type {transition_type} is not available')

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

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
            transition_type=transition_type
        )

    def _compute_opm(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _offload_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m, z = input_tensors

        if (_offload_inference and inplace_safe):
            # m: GPU, z: CPU
            del m, z
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            m, z = input_tensors

        opm = self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: GPU
            del m, z
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        z = add(z, opm, inplace=inplace_safe)
        del opm

        return m, z

    @abstractmethod
    def forward(self,
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


class EvoformerBlock(MSABlock):
    def __init__(self,
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
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        super(EvoformerBlock, self).__init__(c_m=c_m,
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
                                             eps=eps)

        # Specifically, seqemb mode does not use column attention
        self.no_column_attention = no_column_attention

        if not self.no_column_attention:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
            )

    def forward(self,
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

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        m = add(m,
                self.msa_dropout_layer(
                    self.msa_att_row(
                        m,
                        z=z,
                        mask=msa_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                    )
                ),
                inplace=inplace_safe,
                )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: CPU
            del m, z
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors

        # Specifically, column attention is not used in seqemb mode.
        if not self.no_column_attention:
            m = add(m,
                    self.msa_att_col(
                        m,
                        mask=msa_mask,
                        chunk_size=chunk_size,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        use_flash=use_flash,
                    ),
                    inplace=inplace_safe,
                    )

        m = add(
            m,
            self.msa_transition(
                m, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        if not self.opm_first:
            if (not inplace_safe):
                input_tensors = [m, z]

            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        if (_offload_inference and inplace_safe):
            # m: CPU, z: GPU
            del m, z
            assert (sys.getrefcount(input_tensors[0]) == 2)
            device = input_tensors[0].device
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        if (not inplace_safe):
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
            _attn_chunk_size=_attn_chunk_size
        )

        if (_offload_inference and inplace_safe):
            # m: GPU, z: GPU
            device = z.device
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[0] = input_tensors[0].to(device)
            m, _ = input_tensors
        else:
            m = input_tensors[0]

        return m, z


class ExtraMSABlock(MSABlock):
    """
        Almost identical to the standard EvoformerBlock, except in that the
        ExtraMSABlock uses GlobalAttention for MSA column attention and
        requires more fine-grained control over checkpointing. Separated from
        its twin to preserve the TorchScript-ability of the latter.
    """
    def __init__(self,
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
    ):
        super(ExtraMSABlock, self).__init__(c_m=c_m,
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
                                            eps=eps)

        self.ckpt = ckpt

        self.msa_att_col = MSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps,
        )

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(input_tensors=input_tensors,
                                     msa_mask=msa_mask,
                                     chunk_size=chunk_size,
                                     inplace_safe=inplace_safe,
                                     _offload_inference=_offload_inference)

        m = add(m,
            self.msa_dropout_layer(
                self.msa_att_row(
                    m.clone() if torch.is_grad_enabled() else m,
                    z=z.clone() if torch.is_grad_enabled() else z,
                    mask=msa_mask,
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_memory_efficient_kernel=not (use_lma or use_deepspeed_evo_attention),
                    _checkpoint_chunks=
                        self.ckpt if torch.is_grad_enabled() else False,
                )
            ),
            inplace=inplace_safe,
        )

        if (not inplace_safe):
            input_tensors = [m, z]

        del m, z

        def fn(input_tensors):
            m, z = input_tensors

            if (_offload_inference and inplace_safe):
                # m: GPU, z: CPU
                del m, z
                assert (sys.getrefcount(input_tensors[1]) == 2)
                input_tensors[1] = input_tensors[1].cpu()
                torch.cuda.empty_cache()
                m, z = input_tensors

            m = add(m,
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
                    m, mask=msa_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
            )

            if not self.opm_first:
                if (not inplace_safe):
                    input_tensors = [m, z]

                del m, z

                m, z = self._compute_opm(input_tensors=input_tensors,
                                         msa_mask=msa_mask,
                                         chunk_size=chunk_size,
                                         inplace_safe=inplace_safe,
                                         _offload_inference=_offload_inference)

            if (_offload_inference and inplace_safe):
                # m: CPU, z: GPU
                del m, z
                assert (sys.getrefcount(input_tensors[0]) == 2)
                device = input_tensors[0].device
                input_tensors[0] = input_tensors[0].cpu()
                input_tensors[1] = input_tensors[1].to(device)
                m, z = input_tensors

            if (not inplace_safe):
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
                _attn_chunk_size=_attn_chunk_size
            )

            m = input_tensors[0]
            if (_offload_inference and inplace_safe):
                # m: GPU, z: GPU
                device = z.device
                del m
                assert (sys.getrefcount(input_tensors[0]) == 2)
                input_tensors[0] = input_tensors[0].to(device)
                m, _ = input_tensors

            return m, z

        if (torch.is_grad_enabled() and self.ckpt):
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, input_tensors)
        else:
            m, z = fn(input_tensors)

        return m, z


class MSAModuleBlock(EvoformerBlock):
    def __init__(self,
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
        super(MSAModuleBlock, self).__init__(c_m=c_m,
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
                                             eps=eps)

        self.msa_att_row = MSAPairWeightedAveraging(
            c_in=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

