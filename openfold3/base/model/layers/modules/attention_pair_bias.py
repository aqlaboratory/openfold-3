from typing import Optional, List

import torch

from openfold3.base.model.primitives import (AdaLN, LayerNorm, Linear,
                                             Attention, DEFAULT_LMA_Q_CHUNK_SIZE, DEFAULT_LMA_KV_CHUNK_SIZE)

from openfold3.base.utils.tensor_utils import permute_final_dims


class AttentionPairBias(Attention):
    """
    Implements AF3 Algorithm 24.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf=1e9
    ):
        """

        Args:
            c_q:
            c_k:
            c_v:
            c_z:
            c_hidden:
            no_heads:
            use_ada_layer_norm:
            gating:
            inf:
        """
        super(AttentionPairBias, self).__init__(c_q=c_q, c_k=c_k, c_v=c_v, c_hidden=c_hidden,
                                                no_heads=no_heads, gating=gating)

        self.use_ada_layer_norm = use_ada_layer_norm
        self.c_z = c_z
        self.inf = inf

        if self.use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_in=c_q)
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(
            self.c_z, self.no_heads, bias=False, init="normal"
        )

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=True, init="glorot"
        )

        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, bias=False, init="final"
        )

        if self.use_ada_layer_norm:
            self.linear_ada_out = Linear(self.c_q, self.c_q, init="gating_ada_zero")

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        """

        Args:
            a:
            z:
            beta:
            mask:

        Returns:

        """
        if mask is None:
            # [*, I, J]
            mask = a.new_ones(
                a.shape[:-1],
            )

        # [*, N_res, N_res]
        square_mask = mask[..., None] * mask[..., None, :]
        # [*, 1, N_res, N_res]
        mask_bias = (self.inf * (square_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        # [*, N_res, N_res, C_z]
        z = self.layer_norm_z(z)

        # [*, N_res, N_res, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N_res, N_res]
        z = permute_final_dims(z, [2, 0, 1])

        if beta is not None:
            z = z + beta.unsqueeze(-3)

        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE
    ) -> torch.Tensor:
        """

        Args:
            a:
            z:
            s:
            beta:
            mask:
            use_memory_efficient_kernel:
            use_deepspeed_evo_attention:
            use_lma:
            lma_q_chunk_size:
            lma_kv_chunk_size:

        Returns:

        """
        if self.use_ada_layer_norm:
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        biases = self._prep_bias(a, z, beta, mask)

        # Do we support all the memory efficient kernel types?
        a = super(AttentionPairBias, self).forward(
            q_x=a,
            kv_x=a,
            biases=biases,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
        )

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a
