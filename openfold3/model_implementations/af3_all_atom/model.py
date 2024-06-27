from typing import Dict

import torch
from torch import nn
from ml_collections import ConfigDict

from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
from openfold3.core.model.feature_embedders.template_embedders import TemplatePairEmbedderAllAtom
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template import TemplatePairStack
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.structure.diffusion_module import DiffusionModule, SampleDiffusion
from openfold3.core.model.feature_embedders import InputEmbedderAllAtom


class AlphaFold3(nn.Module):

    def __init__(self, config: ConfigDict):
        super(AlphaFold3, self).__init__()
        self.config = config
        self.globals = self.config.globals

        self.input_embedder = InputEmbedderAllAtom(**self.config.model.input_embedder)

        self.layer_norm_z = LayerNorm(self.globals.c_z)
        self.linear_z = Linear(self.globals.c_z, self.globals.c_z, bias=False)

        self.template_pair_embedder = TemplatePairEmbedderAllAtom(**self.config.template_pair_embedder)
        self.template_pair_stack = TemplatePairStack(**self.config.template_pair_stack)

        self.msa_module_embedder = MSAModuleEmbedder(**self.config.model.msa_module_embedder)
        self.msa_module = MSAModuleStack(**self.config.model.msa_module)

        self.layer_norm_s = LayerNorm(self.globals.c_s)
        self.linear_s = Linear(self.globals.c_s, self.globals.c_s, bias=False)

        self.pairformer_stack = PairFormerStack(**self.config.model.pairformer)

        self.diffusion_module = DiffusionModule(config=self.config.model.diffusion_module)
        
        self.sample_diffusion = SampleDiffusion(**self.config.model.sample_diffusion,
                                                diffusion_module=self.diffusion_module)

        self.confidence_head = None

        self.distogram_head = None

    def _encode(self, batch):

        s_input, s_init, z_init = self.input_embedder(batch)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for _ in range(self.globals.no_cycles):

            z = z_init + self.linear_z(self.layer_norm_z(z))
            z = z + self.template_embedder(batch, z)
            z = z + self.msa_module(batch, z, s_input)

            s = s_init + self.linear_s(self.layer_norm_s(s))
            s, z = self.pairformer_stack(s, z)

        return s_input, s, z

    def _rollout(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> Dict:

        # Compute atom positions
        with torch.no_grad():
            x_pred = self.sample_diffusion(batch=batch,
                                           si_input=si_input,
                                           si_trunk=si_trunk,
                                           zij_trunk=zij_trunk)

        # Compute confidences
        p_plddt, p_pae, p_pde, p_resolved = self.confidence_head(si_input, si_trunk, zij_trunk, x_pred)

        # Compute distogram
        p_distogram = self.distogram_head(zij_trunk)

        return {
            'x_pred': x_pred,
            'p_plddt': p_plddt,
            'p_pae': p_pae,
            'p_pde': p_pde,
            'p_resolved': p_resolved,
            'p_distogram': p_distogram
        }

    def _train(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        xl_gt: torch.Tensor
    ) -> torch.Tensor:

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
        noise = (t[:, None, None] ** 2) * torch.randn((batch_size, no_samples, n_atom, 3), device=device)

        # Sample atom positions
        xl_noisy = xl_gt + noise

        # Run diffusion module
        xl = self.diffusion_module(batch=batch,
                                   xl_noisy=xl_noisy,
                                   t=t,
                                   si_input=si_input,
                                   si_trunk=si_trunk,
                                   zij_trunk=zij_trunk)
        
        return xl

    def forward(self, batch, xl_gt=None):

        # Compute representations
        si_input, si_trunk, zij_trunk = self._encode(batch)

        # [b, ...]

        # Mini rollout
        output = self._rollout(batch=batch,
                               si_input=si_input,
                               si_trunk=si_trunk,
                               zij_trunk=zij_trunk)

        # Run training step (if necessary)
        if self.training:

            xl = self._train(batch=batch,
                            si_input=si_input,
                            si_trunk=si_trunk,
                            zij_trunk=zij_trunk,
                            xl_gt=xl_gt)

            return output, xl

        return output
