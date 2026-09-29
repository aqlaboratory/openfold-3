# Copyright 2026 AlQuraishi Laboratory
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

import unittest

import torch

from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import randomize_parameters


class TestDiffusionConditioning(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
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
            "cyclic_mask": torch.zeros((batch_size, 1, n_token), dtype=torch.bool),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
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

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
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
            "cyclic_mask": torch.zeros((batch_size, 1, n_token), dtype=torch.bool),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (batch_size, n_sample, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))

    def test_with_same_schedule(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
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
            "cyclic_mask": torch.zeros((batch_size, 1, n_token), dtype=torch.bool),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (batch_size, 1, n_token, c_s))
        self.assertTrue(zij.shape == (batch_size, 1, n_token, n_token, c_z))

    def test_precomputed_pair_conditioning(self):
        """
        `condition_pair` is the pair-conditioning half of `forward`, extracted so it
        can be precomputed once per diffusion rollout (it does not depend on the
        noise level `t`). Passing its result to `forward` as `zij` must give the
        exact same output as calling `forward` without it, for every noise level.
        """
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        n_sample = 3

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})

        dc = randomize_parameters(DiffusionConditioning(**diff_cond_config).eval())

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

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
            "cyclic_mask": torch.zeros((batch_size, 1, n_token), dtype=torch.bool),
        }
        batch["token_mask"][..., -3:] = 0

        # The pair conditioning must not depend on the noise level, so a single
        # precomputed pair representation is reused for all of these
        t_steps = [
            diff_cond_config.sigma_data
            * torch.exp(-1.2 + 1.5 * torch.randn((batch_size, n_sample)))
            for _ in range(2)
        ]

        for use_conditioning in (True, False):
            for chunk_size in (None, 4):
                with (
                    self.subTest(use_conditioning=use_conditioning, chunk=chunk_size),
                    torch.no_grad(),
                ):
                    zij_pre = dc.condition_pair(
                        batch=batch,
                        zij_trunk=zij_trunk,
                        use_conditioning=use_conditioning,
                        chunk_size=chunk_size,
                    )

                    for t in t_steps:
                        si_expected, zij_expected = dc(
                            batch=batch,
                            t=t,
                            si_input=si_input,
                            si_trunk=si_trunk,
                            zij_trunk=zij_trunk,
                            use_conditioning=use_conditioning,
                            chunk_size=chunk_size,
                        )
                        si, zij = dc(
                            batch=batch,
                            t=t,
                            si_input=si_input,
                            si_trunk=si_trunk,
                            zij_trunk=None,
                            use_conditioning=use_conditioning,
                            chunk_size=chunk_size,
                            zij=zij_pre,
                        )

                        self.assertIs(zij, zij_pre)
                        torch.testing.assert_close(
                            zij_pre, zij_expected, rtol=1e-6, atol=1e-6
                        )
                        torch.testing.assert_close(
                            si, si_expected, rtol=1e-6, atol=1e-6
                        )


if __name__ == "__main__":
    unittest.main()
