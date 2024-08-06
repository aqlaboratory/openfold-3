import unittest

import torch

from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.model_implementations.af3_all_atom.config.base_config import config
from tests.config import consts


class TestDiffusionConditioning(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        diff_cond_config = config.model.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn(batch_size, device=si_trunk.device)
        )

        batch = {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(si.shape == (batch_size, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, n_token, n_token, c_z))

    def test_with_different_schedule(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        n_sample = 3

        diff_cond_config = config.model.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn((batch_size, n_sample), device=si_trunk.device)
        )

        batch = {
            "token_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "token_mask": torch.ones((batch_size, 1, n_token)),
            "residue_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "sym_id": torch.zeros((batch_size, 1, n_token)),
            "asym_id": torch.zeros((batch_size, 1, n_token)),
            "entity_id": torch.zeros((batch_size, 1, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(si.shape == (batch_size, n_sample, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))

    def test_with_same_schedule(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        diff_cond_config = config.model.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = DiffusionConditioning(**diff_cond_config)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        t = diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn((1, 1), device=si_trunk.device)
        )

        batch = {
            "token_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "token_mask": torch.ones((batch_size, 1, n_token)),
            "residue_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "sym_id": torch.zeros((batch_size, 1, n_token)),
            "asym_id": torch.zeros((batch_size, 1, n_token)),
            "entity_id": torch.zeros((batch_size, 1, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(si.shape == (batch_size, 1, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))


if __name__ == "__main__":
    unittest.main()
