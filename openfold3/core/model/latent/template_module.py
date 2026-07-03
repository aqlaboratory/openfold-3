# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2025 NVIDIA Corporation
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

"""Template embedding layers.

These modules embed templates into pair embeddings. Note that this includes the template
feature embedding functions in openfold3.core.model.feature_embedders.
"""

from functools import partial

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.core.model.latent.base_blocks import PairBlock
from openfold3.core.model.layers.transition import SwiGLUTransition
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.primitives.fused_swiglu_transition import (
    is_fused_swiglu_transition_enabled,
)
from openfold3.core.model.utils import assert_sole_holder
from openfold3.core.utils.checkpointing import checkpoint_blocks, checkpoint_section
from openfold3.core.utils.chunk_utils import (
    DEFAULT_MAX_CHUNK_SIZE,
    FLASH_MAX_CHUNK_SIZE,
    ChunkSizeTuner,
    apply_triangle_attn_chunk_cap,
)
from openfold3.core.utils.tensor_utils import add

_TEMPLATE_DIM_BY_KEY = {
    "template_restype": -3,
    "template_pseudo_beta_mask": -2,
    "template_backbone_frame_mask": -2,
    "template_distogram": -4,
    "template_unit_vector": -4,
}


def _slice_template_feature(k: str, x: torch.Tensor, idx: int) -> torch.Tensor:
    """Return one template while preserving the singleton template dimension."""
    template_dim = _TEMPLATE_DIM_BY_KEY.get(k)
    if template_dim is None:
        return x

    slices = [slice(None)] * x.dim()
    slices[template_dim] = slice(idx, idx + 1)
    return x[tuple(slices)]


# TODO: Make arguments match PairBlock
class TemplatePairBlock(PairBlock):
    """Implements one block of AF2 Algorithm 16."""

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        no_heads: int,
        transition_type: str,
        pair_transition_n: int,
        dropout_rate: float,
        tri_mul_first: bool,
        fuse_projection_weights: bool,
        ckpt_per_template: bool,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pair_block_init,
        use_reentrant: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_mul:
                Hidden dimension for triangular multiplication
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            no_heads:
                Number of heads in the attention mechanism
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            tri_mul_first:
                Whether to perform triangular multiplication before attention
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            ckpt_per_template:
                Whether to use activation checkpointing per template
            inf:
                Large constant used for masking
            linear_init_params:
                Configuration for linear initialization
        """
        super().__init__(
            c_z=c_t,
            c_hidden_mul=c_hidden_tri_mul,
            c_hidden_pair_att=c_hidden_tri_att,
            no_heads_pair=no_heads,
            transition_type=transition_type,
            transition_n=pair_transition_n,
            pair_dropout=dropout_rate,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            linear_init_params=linear_init_params,
        )

        self.tri_mul_first = tri_mul_first
        self.ckpt_per_template = ckpt_per_template
        self.use_reentrant = use_reentrant

    def _forward_single_template(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int | None,
        use_deepspeed_evo_attention: bool,
        use_cueq_triangle_kernels: bool,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ):
        """
        Helper function to process exactly one template slice.
        """

        # t: [*, 1, N, N, C]. Collapse the singleton template axis during
        # inference so the 4D fused triangle kernels can handle template blocks
        # the same way they handle trunk/MSA pair blocks.
        squeezed_template_dim = False
        if (
            inplace_safe
            and not self.training
            and not torch.is_grad_enabled()
            and t.shape[-4] == 1
            and mask.shape[-3] == 1
        ):
            t = t.squeeze(-4)
            mask = mask.squeeze(-3)
            squeezed_template_dim = True

        if self.tri_mul_first:
            t = self.tri_att_start_end(
                z=self.tri_mul_out_in(
                    z=t,
                    pair_mask=mask,
                    inplace_safe=inplace_safe,
                    use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                    use_triton_triangle_kernels=use_triton_triangle_kernels,
                ),
                _attn_chunk_size=_attn_chunk_size,
                pair_mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            t = self.tri_mul_out_in(
                z=self.tri_att_start_end(
                    z=t,
                    _attn_chunk_size=_attn_chunk_size,
                    pair_mask=mask,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                    use_triton_triangle_kernels=use_triton_triangle_kernels,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
                pair_mask=mask,
                inplace_safe=inplace_safe,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
            )

        if squeezed_template_dim:
            t = t.unsqueeze(-4)
            mask = mask.unsqueeze(-3)

        pair_trans_mask = mask if _mask_trans else None
        # Same fused pair-transition fast path as PairBlock in base_blocks.py.
        if (
            is_fused_swiglu_transition_enabled()
            and inplace_safe
            and not torch.is_grad_enabled()
            and isinstance(self.pair_transition, SwiGLUTransition)
        ):
            trans_mask = (
                pair_trans_mask.unsqueeze(-1) if pair_trans_mask is not None else None
            )
            t = self.pair_transition._transition_inplace(
                x=t, mask=trans_mask, residual=t
            )
        else:
            t = add(
                t,
                self.pair_transition(
                    t,
                    mask=pair_trans_mask,
                    chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
            )

        return t

    def forward(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] Template embedding
            mask:
                [*, N_templ, N_res, N_res] Template mask
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers
            _attn_chunk_size:
                Inference-time subbatch size for attention. If None, uses chunk.

        Returns:
            [*, N_templ, N_res, N_res, C_t] Template embedding update
        """

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        single_templates = [t_i.unsqueeze(-4) for t_i in torch.unbind(t, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]

        apply_ckpt = self.training and self.ckpt_per_template
        single_templ_fn = partial(
            self._forward_single_template,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        for i in range(len(single_templates)):
            t_in = single_templates[i]
            mask_in = single_templates_masks[i]

            # Make contiguous to avoid activation checkpointing on a views of tensors
            # If inplace_safe is true (inference), avoid making a copy of the tensor
            if not inplace_safe and apply_ckpt:
                t_in = t_in.contiguous()
                mask_in = mask_in.contiguous()

            t_out = checkpoint_section(
                single_templ_fn,
                args=(t_in, mask_in),
                apply_ckpt=apply_ckpt,
                use_reentrant=self.use_reentrant,
            )

            if inplace_safe:
                # t_in is the view into t at index i
                # Copy here to safely update t if t_out has any non-inplace updates
                t_in.copy_(t_out)
            else:
                single_templates[i] = t_out

        if not inplace_safe:
            t = torch.cat(single_templates, dim=-4)

        return t


class TemplatePairStack(nn.Module):
    """Implements AF2 Algorithm 16."""

    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        transition_type,
        pair_transition_n,
        dropout_rate,
        tri_mul_first,
        fuse_projection_weights,
        blocks_per_ckpt,
        ckpt_per_template,
        inf=1e9,
        linear_init_params=lin_init.pair_block_init,
        use_reentrant: bool | None = None,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_mul:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            no_heads:
                Number of heads in the attention mechanism
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            tri_mul_first:
                Whether to perform triangular multiplication before attention
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
            ckpt_per_template:
                Whether to do activation checkpointing per template.
                This will disable the per-block checkpointing.
            inf:
                Large constant used for masking
            linear_init_params:
                Configuration for linear initialization
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
            tune_chunk_size:
                 Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.blocks_per_ckpt = None if ckpt_per_template else blocks_per_ckpt
        self.use_reentrant = use_reentrant

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_mul=c_hidden_tri_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                no_heads=no_heads,
                transition_type=transition_type,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                tri_mul_first=tri_mul_first,
                fuse_projection_weights=fuse_projection_weights,
                ckpt_per_template=ckpt_per_template,
                inf=inf,
                linear_init_params=linear_init_params,
                use_reentrant=use_reentrant,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            use_flash_kernels = (
                use_cueq_triangle_kernels
                or use_triton_triangle_kernels
                or use_deepspeed_evo_attention
            )
            max_chunk_size = (
                FLASH_MAX_CHUNK_SIZE if use_flash_kernels else DEFAULT_MAX_CHUNK_SIZE
            )
            n_tokens = t.shape[-3]
            capped_chunk_size = apply_triangle_attn_chunk_cap(
                chunk_size, max_chunk_size, n_tokens=n_tokens
            )
            if capped_chunk_size is not None:
                tuned_chunk_size = capped_chunk_size
            else:
                tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                    representative_fn=blocks[0],
                    args=(t.clone(),),
                    min_chunk_size=chunk_size,
                    max_chunk_size=max_chunk_size,
                )
            attn_chunk = (
                tuned_chunk_size if use_flash_kernels else (tuned_chunk_size // 4)
            )
            attn_chunk = max(chunk_size, attn_chunk)
            capped_attn_chunk = apply_triangle_attn_chunk_cap(
                chunk_size, attn_chunk, n_tokens=n_tokens
            )
            if capped_attn_chunk is not None:
                attn_chunk = capped_attn_chunk
            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    _attn_chunk_size=attn_chunk,
                )
                for b in blocks
            ]

        return blocks

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
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
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers

        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = self._prep_blocks(
            t=t,
            mask=mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        (t,) = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
            use_reentrant=self.use_reentrant,
        )

        t = self.layer_norm(t)

        return t


class TemplateEmbedderAllAtom(nn.Module):
    """Implements AF3 Algorithm 16."""

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                ConfigDict with template config.
        """
        super().__init__()

        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderAllAtom(
            **config.template_pair_embedder,
        )
        self.template_pair_stack = TemplatePairStack(
            **config.template_pair_stack,
        )

        templ_init = config.get(
            "linear_init_params", lin_init.all_atom_templ_module_init
        )
        self.linear_t = Linear(config.c_t, config.c_z, **templ_init.linear_t)

    def _forward_offload(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_token, N_token, C_z] Pair embedding
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size.
            _mask_trans:
                Whether to mask the output of the transition layers
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            t:
                [*, N_token, N_token, C_z] Template embedding
        """
        batch_dims = z.shape[:-3]
        n_token = z.shape[-2]
        out_device = z.device
        n_templ = batch["template_restype"].shape[-3]

        # [*, 1, N_token, N_token]
        pair_mask = pair_mask[..., None, :, :].to(dtype=z.dtype)

        t_out = torch.zeros(
            (*batch_dims, n_templ, n_token, n_token, self.config.c_t), device="cpu"
        )

        for i in range(n_templ):
            batch_templ = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k.startswith("template_"):
                    sliced = _slice_template_feature(k, v, i)
                    if sliced.device != out_device:
                        sliced = sliced.to(out_device, non_blocking=True)
                    batch_templ[k] = sliced
                else:
                    batch_templ[k] = v

            # [*, N, N, C_t]
            t = self.template_pair_embedder(
                batch=batch_templ,
                z=z,
            )

            # [*, N_templ, N_token, N_token, C_z]
            t = self.template_pair_stack(
                t,
                pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

            assert_sole_holder(t)

            t_out[..., i : i + 1, :, :, :] = t.cpu()

            del t

        del z

        return t_out.to(device=out_device)

    # by Liang Hong <lhong22@cse.cuhk.edu.hk>: per-template streaming
    # template stack with fused pair embedder dispatch (n_templ=1).
    def _forward_streaming(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Inference-only GPU streaming path.

        Processes one template at a time and accumulates the post-stack
        template sum on GPU.  This avoids materializing the full
        ``[N_templ, N, N, c_t]`` stack without paying the CPU offload copy.

        Any ``batch["template_*"]`` tensor that already lives on CPU (the
        caller may have offloaded it at the top of ``OpenFold3.forward``
        to keep the raw ~1.2U ``template_distogram`` out of GPU HBM) is
        streamed H→D one template at a time by the existing
        ``sliced.to(out_device, non_blocking=True)`` copy below.  No
        special-casing needed here — the branch handles both regimes.
        """
        n_templ = batch["template_restype"].shape[-3]
        out_device = z.device

        pair_mask = pair_mask[..., None, :, :].to(dtype=z.dtype)
        t_sum = None

        for i in range(n_templ):
            batch_templ = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k.startswith("template_"):
                    sliced = _slice_template_feature(k, v, i)
                    if sliced.device != out_device:
                        sliced = sliced.to(out_device, non_blocking=True)
                    batch_templ[k] = sliced
                else:
                    batch_templ[k] = v

            t = self.template_pair_embedder(batch=batch_templ, z=z)
            # Release the sliced [B, 1, N, N, 39] template_distogram (~0.30U at
            # N=1264 fp32) and [B, 1, N, N, 3] template_unit_vector immediately
            # after the embedder has folded them into ``t``.  Otherwise they
            # coexist with the pair_stack peak below, lifting the template phase
            # transient by ~0.30U per iteration.
            batch_templ.pop("template_distogram", None)
            batch_templ.pop("template_unit_vector", None)
            t = self.template_pair_stack(
                t,
                pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

            t = t.squeeze(-4)
            if t_sum is None:
                t_sum = t
            else:
                t_sum.add_(t)
            del t

        assert t_sum is not None
        return t_sum / n_templ

    def _forward(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # [*, N_templ, N_token, N_token, C_t]
        t = self.template_pair_embedder(batch, z)

        # [*, 1, N_token, N_token]
        pair_mask = pair_mask[..., None, :, :].to(dtype=z.dtype)

        # [*, N_templ, N_token, N_token, C_z]
        t = self.template_pair_stack(
            t,
            pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        return t

    def forward(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        offload_inference: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_token, N_token, C_z] Pair embedding
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size.
            _mask_trans:
                Whether to mask the output of the transition layers
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_cueq_triangle_kernels:
                Whether to use cuEq triangle kernels
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            offload_inference:
                Whether to offload some computation to CPU

        Returns:
            t:
                [*, N_token, N_token, C_z] Template embedding
        """
        n_templ = batch["template_restype"].shape[-3]
        template_sum_done = False
        can_stream = (
            inplace_safe
            and not self.training
            and not torch.is_grad_enabled()
            and n_templ > 1
        )
        if can_stream:
            t = self._forward_streaming(
                batch=batch,
                z=z,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=True,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
            template_sum_done = True
        elif offload_inference:
            t = self._forward_offload(
                batch=batch,
                z=z,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=True,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            t = self._forward(
                batch=batch,
                z=z,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=True,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )

        if not template_sum_done:
            t = torch.sum(t, dim=-4) / n_templ

        # [*, N_token, N_token, C_z]
        if not torch.is_grad_enabled():
            t = t.relu_()
        else:
            t = torch.nn.functional.relu(t)
        t = self.linear_t(t)

        return t
