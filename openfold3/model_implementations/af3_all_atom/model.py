from typing import Dict, Tuple

import torch
from ml_collections import ConfigDict
from torch import nn

from openfold3.core.model.feature_embedders import InputEmbedderAllAtom
from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
)
from openfold3.core.utils.tensor_utils import add


class AlphaFold3(nn.Module):
    def __init__(self, config: ConfigDict):
        """

        Args:
            config:
        """
        super().__init__()
        self.config = config
        self.globals = self.config.globals

        self.input_embedder = InputEmbedderAllAtom(**self.config.model.input_embedder)

        self.layer_norm_z = LayerNorm(self.globals.c_z)
        self.linear_z = Linear(self.globals.c_z, self.globals.c_z, bias=False)

        self.template_embedder = TemplateEmbedderAllAtom(
            config=self.config.model.template
        )

        self.msa_module_embedder = MSAModuleEmbedder(
            **self.config.model.msa.msa_module_embedder
        )
        self.msa_module = MSAModuleStack(**self.config.model.msa.msa_module)

        self.layer_norm_s = LayerNorm(self.globals.c_s)
        self.linear_s = Linear(self.globals.c_s, self.globals.c_s, bias=False)

        self.pairformer_stack = PairFormerStack(**self.config.model.pairformer)

        self.diffusion_module = DiffusionModule(
            config=self.config.model.diffusion_module
        )

        self.sample_diffusion = SampleDiffusion(
            **self.config.model.sample_diffusion, diffusion_module=self.diffusion_module
        )

        self.confidence_head = None

        self.distogram_head = None

    def _disable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.msa_module.blocks_per_ckpt = None
        self.pairformer_stack.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.msa_module.blocks_per_ckpt = self.config.msa.msa_module.blocks_per_ckpt
        self.pairformer_stack.blocks_per_ckpt = self.config.pairformer.blocks_per_ckpt

    def run_trunk(
        self, batch: Dict, inplace_safe: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            batch:
            inplace_safe:
                Whether inplace operations can be performed

        Returns:

        """
        s_input, s_init, z_init = self.input_embedder(batch=batch)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # token_mask: [*, N_token]
        # pair_mask: [*, N_token, N_token]
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        for _ in range(self.globals.no_cycles):
            # [*, N_token, N_token, C_z]
            z = z_init + self.linear_z(self.layer_norm_z(z))

            z = add(
                z,
                self.template_embedder(
                    batch=batch,
                    z=z,
                    pair_mask=pair_mask,
                    chunk_size=self.globals.chunk_size,
                    _mask_trans=True,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
            )

            m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)

            # Run MSA + pair embeddings through the MsaModule
            # m: [*, N_seq, N_token, C_m]
            # z: [*, N_token, N_token, C_z]
            if self.globals.offload_inference:
                input_tensors = [m, z]
                del m, z
                z = self.msa_module.forward_offload(
                    input_tensors,
                    msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                    pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    _mask_trans=True,
                )

                del input_tensors
            else:
                z = self.msa_module(
                    m,
                    z,
                    msa_mask=msa_mask.to(dtype=m.dtype),
                    pair_mask=pair_mask.to(dtype=z.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    inplace_safe=inplace_safe,
                    _mask_trans=True,
                )

            del m, msa_mask

            s = s_init + self.linear_s(self.layer_norm_s(s))
            s, z = self.pairformer_stack(
                s=s,
                z=z,
                single_mask=token_mask.to(dtype=z.dtype),
                pair_mask=pair_mask.to(dtype=s.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=True,
            )

        del s_init, z_init

        return s_input, s, z

    def _rollout(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> Dict:
        """

        Args:
            batch:
            si_input:
            si_trunk:
            zij_trunk:

        Returns:

        """
        # Compute atom positions
        with torch.no_grad():
            x_pred = self.sample_diffusion(
                batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk
            )

        # # Compute confidences
        # p_plddt, p_pae, p_pde, p_resolved = self.confidence_head(
        #     si_input, si_trunk, zij_trunk, x_pred
        # )
        #
        # # Compute distogram
        # p_distogram = self.distogram_head(zij_trunk)

        return {
            "x_pred": x_pred,
            # "p_plddt": p_plddt,
            # "p_pae": p_pae,
            # "p_pde": p_pde,
            # "p_resolved": p_resolved,
            # "p_distogram": p_distogram,
        }

    def _train(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        xl_gt: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            batch:
            si_input:
            si_trunk:
            zij_trunk:
            xl_gt:

        Returns:

        """
        # Expand sampling dimension
        # Is this ideal?
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)
        xl_gt = xl_gt.unsqueeze(1)
        for key in batch:
            batch[key] = batch[key].unsqueeze(1)

        # Sample noise schedule for training
        no_samples = self.globals.no_samples
        batch_size, n_atom, device = xl_gt.shape[0], xl_gt.shape[1], xl_gt.device
        n = torch.randn((batch_size, no_samples), device=device)
        t = self.globals.sigma_data * torch.exp(-1.2 + 1.5 * n)

        # Sample noise
        noise = (t[..., None, None] ** 2) * torch.randn(
            (batch_size, no_samples, n_atom, 3), device=device
        )

        # Sample atom positions
        xl_noisy = xl_gt + noise

        # Run diffusion module
        xl = self.diffusion_module(
            batch=batch,
            xl_noisy=xl_noisy,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        return xl

    def forward(self, batch: Dict, xl_gt: torch.Tensor = None) -> Dict:
        """

        Args:
            batch:
            xl_gt:

        Returns:

        """
        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in batch:
            if batch[k].dtype == torch.float32:
                batch[k] = batch[k].to(dtype=dtype)

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Compute representations
        si_input, si_trunk, zij_trunk = self.run_trunk(
            batch=batch, inplace_safe=inplace_safe
        )

        # Mini rollout
        output = self._rollout(
            batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk
        )

        # Run training step (if necessary)
        if self.training:
            xl = self._train(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                xl_gt=xl_gt,
            )

            output["x_train"] = xl

        return output
