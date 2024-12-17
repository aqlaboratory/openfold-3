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

"""
Sequence-local atom attention modules. Includes AtomAttentionEncoder,
AtomAttentionDecoder, and AtomTransformer.
"""

from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.atomize_utils import (
    aggregate_atom_feat_to_tokens,
    broadcast_token_feat_to_atoms,
)
from openfold3.core.utils.checkpointing import checkpoint_section

TensorDict = dict[str, torch.Tensor]


def compute_neighborhood_mask(
    n_query: int,
    n_key: int,
    n_atom: int,
    create_sparsity_mask: bool,
    block_size: Optional[int],
    no_batch_dims: int,
    device: torch.device,
    dtype: torch.dtype,
    inf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute neighborhood mask for Sequence-local atom attention.

    Args:
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        n_atom:
            Number of atoms
        create_sparsity_mask:
            Whether to create the sparsity layout mask used in block sparse attention
        block_size:
            Block size to use in block sparse attention
        no_batch_dims:
            Number of batch dimensions
        device:
            Device to use
        dtype:
            Dtype to use
        inf:
            Large number used for attention masking

    Returns:
        beta:
            [*, N_atom, N_atom] Atom neighborhood mask
        layout:
            # [N_atom / block_size, N_atom / block_size] Sparsity layout mask
    """
    if create_sparsity_mask:
        n_query = n_query // block_size
        n_key = n_key // block_size
        n_blocks = n_atom // block_size
    else:
        n_query = n_query
        n_key = n_key
        n_blocks = n_atom

    offset = n_query // 2 - 0.5
    n_center = int(n_blocks // n_query) + 1

    # Define subset centers
    # [N_center]
    subset_centers = offset + torch.arange(n_center, device=device) * n_query

    # If use_block_sparse_attn: [N_atom / block_size, N_query / block_size]
    # Else: [N_atom, N_query]
    row_mask = torch.abs(
        torch.arange(n_blocks, device=device).unsqueeze(1) - subset_centers.unsqueeze(0)
    ) < (n_query / 2)

    # If use_block_sparse_attn: [N_atom / block_size, N_key / block_size]
    # Else: [N_atom, N_key]
    col_mask = torch.abs(
        torch.arange(n_blocks, device=device).unsqueeze(1) - subset_centers.unsqueeze(0)
    ) < (n_key / 2)

    # Compute beta
    # If use_block_sparse_attn: [N_atom / block_size, N_atom / block_size]
    # Else: [N_atom, N_atom]
    beta = torch.einsum("li,mi->lm", row_mask.to(dtype), col_mask.to(dtype))

    layout = None
    if create_sparsity_mask:
        # [N_atom / block_size, N_atom / block_size]
        layout = beta
        # [N_atom, N_atom]
        beta = beta.repeat_interleave(block_size, dim=0).repeat_interleave(
            block_size, dim=1
        )

    beta = (beta - 1.0) * inf

    # [*, N_atom, N_atom]
    beta = beta.reshape(no_batch_dims * (1,) + (n_atom, n_atom)).to(device)

    return beta, layout


class AtomTransformer(nn.Module):
    """
    Atom Transformer: neighborhood-blocked (32 * 128) diffusion transformer.

    Implements AF3 Algorithm 7.
    """

    def __init__(
        self,
        c_q: int,
        c_p: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        use_ada_layer_norm: bool = True,
        use_block_sparse_attn: bool = False,
        block_size: Optional[int] = 16,
        blocks_per_ckpt: Optional[int] = None,
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.atom_transformer_init,
        use_reentrant: Optional[bool] = None,
    ):
        """
        Args:
            c_q:
                Atom single representation channel dimension
            c_p:
                Atom pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            use_block_sparse_attn:
                Whether to use Triton block sparse attention kernels
            block_size:
                Block size to use in block sparse attention
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            inf:
                Large number used for attention masking
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.use_block_sparse_attn = use_block_sparse_attn
        self.block_size = block_size
        self.inf = inf

        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_q,
            c_s=c_q,
            c_z=c_p,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            use_block_sparse_attn=self.use_block_sparse_attn,
            block_size=self.block_size,
            blocks_per_ckpt=blocks_per_ckpt,
            inf=self.inf,
            linear_init_params=linear_init_params.diffusion_transformer,
            use_reentrant=use_reentrant,
        )

    def forward(
        self,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        atom_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: Optional[bool] = False,
    ):
        """
        Args:
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
            atom_mask:
                [*, N_atom] Atom mask
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
        Returns:
            ql:
                [*, N_atom, c_atom] Updated atom single representation
        """
        pad_len = 0
        if self.use_block_sparse_attn:
            pad_len = (
                self.block_size - cl.shape[-2] % self.block_size
            ) % self.block_size
            if pad_len > 0:
                ql = torch.nn.functional.pad(ql, (0, 0, 0, pad_len), value=0.0)
                cl = torch.nn.functional.pad(cl, (0, 0, 0, pad_len), value=0.0)
                plm = torch.nn.functional.pad(
                    plm, (0, 0, 0, pad_len, 0, pad_len), value=0.0
                )
                atom_mask = torch.nn.functional.pad(atom_mask, (0, pad_len), value=0.0)

        n_atom = ql.shape[-2]

        # Create atom neighborhood mask
        # beta: [*, N_atom, N_atom]
        # layout: [N_atom / block_size, N_atom / block_size]
        beta, layout = compute_neighborhood_mask(
            n_query=self.n_query,
            n_key=self.n_key,
            n_atom=n_atom,
            create_sparsity_mask=self.use_block_sparse_attn,
            block_size=self.block_size,
            no_batch_dims=len(ql.shape[:-2]),
            device=ql.device,
            dtype=ql.dtype,
            inf=self.inf,
        )

        # Run diffusion transformer
        # [*, N_atom, c_atom]
        ql = self.diffusion_transformer(
            a=ql,
            s=cl,
            z=plm,
            mask=atom_mask,
            beta=beta,
            layout=layout,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
        )

        if pad_len > 0:
            ql = ql[..., :-pad_len, :]

        return ql


class AtomAttentionEncoder(nn.Module):
    """
    Implements AF3 Algorithm 5.
    """

    def __init__(
        self,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        add_noisy_pos: bool,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        use_ada_layer_norm: bool,
        use_block_sparse_attn: bool,
        block_size: Optional[int] = 16,
        blocks_per_ckpt: Optional[int] = None,
        ckpt_intermediate_steps: bool = False,
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.atom_att_enc_init,
        use_reentrant: Optional[bool] = None,
    ):
        """
        Args:
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            add_noisy_pos:
                Whether to add noisy positions and trunk embeddings
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            use_block_sparse_attn:
                Whether to use Triton block sparse attention kernels
            block_size:
                Block size to use in block sparse attention
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            ckpt_intermediate_steps:
                Whether to checkpoint intermediate steps in the module, including
                RefAtomFeatureEmbedder, NoisyPositionEmbedder, and feature aggregation
            inf:
                Large number used for attention masking
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()
        self.add_noisy_pos = add_noisy_pos
        self.ckpt_intermediate_steps = ckpt_intermediate_steps
        self.use_reentrant = use_reentrant

        if self.add_noisy_pos:
            self.linear_r = Linear(3, c_atom, **linear_init_params.linear_r)

        self.atom_transformer = AtomTransformer(
            c_q=c_atom,
            c_p=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            use_ada_layer_norm=use_ada_layer_norm,
            use_block_sparse_attn=use_block_sparse_attn,
            block_size=block_size,
            blocks_per_ckpt=blocks_per_ckpt,
            inf=inf,
            linear_init_params=linear_init_params.atom_transformer,
            use_reentrant=use_reentrant,
        )

        self.c_token = c_token
        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, **linear_init_params.linear_q), nn.ReLU()
        )

    def forward(
        self,
        batch: TensorDict,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        atom_mask: torch.Tensor,
        rl: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: Optional[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
        Returns:
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
        """
        if self.add_noisy_pos:
            # Add noisy coordinate projection
            # [*, N_atom, c_atom]
            ql = ql + self.linear_r(rl)

        # Cross attention transformer (line 15)
        # [*, N_atom, c_atom]
        ql = self.atom_transformer(
            ql=ql,
            cl=cl,
            plm=plm,
            atom_mask=atom_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
        )

        agg_args = (
            batch["token_mask"],
            batch["atom_to_token_index"],
            atom_mask,
            self.linear_q(ql),
            -2,
        )
        ai = checkpoint_section(
            fn=aggregate_atom_feat_to_tokens,
            args=agg_args,
            apply_ckpt=self.ckpt_intermediate_steps,
            use_reentrant=self.use_reentrant,
        )

        return ai, ql, cl, plm


class AtomAttentionDecoder(nn.Module):
    """
    Implements AF3 Algorithm 6.
    """

    def __init__(
        self,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        use_ada_layer_norm: bool,
        use_block_sparse_attn: bool,
        block_size: Optional[int] = 16,
        blocks_per_ckpt: Optional[int] = None,
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.atom_att_dec_init,
        use_reentrant: Optional[bool] = None,
    ):
        """
        Args:
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            use_block_sparse_attn:
                Whether to use Triton block sparse attention kernels
            block_size:
                Block size to use in block sparse attention
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            inf:
                Large number used for attention masking
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()

        self.linear_q_in = Linear(c_token, c_atom, **linear_init_params.linear_q_in)

        self.atom_transformer = AtomTransformer(
            c_q=c_atom,
            c_p=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            use_ada_layer_norm=use_ada_layer_norm,
            use_block_sparse_attn=use_block_sparse_attn,
            block_size=block_size,
            blocks_per_ckpt=blocks_per_ckpt,
            inf=inf,
            linear_init_params=linear_init_params.atom_transformer,
            use_reentrant=use_reentrant,
        )

        self.layer_norm = LayerNorm(c_in=c_atom)
        self.linear_q_out = Linear(c_atom, 3, **linear_init_params.linear_q_out)

    def forward(
        self,
        batch: TensorDict,
        atom_mask: torch.Tensor,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
        Returns:
            rl_update:
                [*, N_atom, 3] Atom position updates
        """
        # Broadcast per-token activations to atoms
        # [*, N_atom, c_atom]
        ql = ql + broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=self.linear_q_in(ai),
            token_dim=-2,
        )

        # Atom transformer
        # [*, N_atom, c_atom]
        ql = self.atom_transformer(
            ql=ql,
            cl=cl,
            plm=plm,
            atom_mask=atom_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
        )

        # Compute updates for atom positions
        # [*, N_atom, 3]
        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update
