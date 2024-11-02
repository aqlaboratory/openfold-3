# Copyright 2021 AlQuraishi Laboratory
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

from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)
from openfold3.projects import registry
from tests.config import consts


class TestDiffusionModule(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res

        proj_entry = registry.get_project_entry("af3_all_atom")
        proj_config = proj_entry.get_config_with_preset()
        config = proj_config.model

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)

        xl_noisy = torch.randn((batch_size, n_atom, 3))
        t = torch.ones(1)
        atom_mask = torch.ones((batch_size, n_atom))
        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        batch = {
            "token_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 119)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
            "num_atoms_per_token": torch.ones((batch_size, n_token)) * 4,
        }

        xl = dm(
            batch=batch,
            xl_noisy=xl_noisy,
            t=t,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(xl.shape == (batch_size, n_atom, 3))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        n_sample = 3

        proj_entry = registry.get_project_entry("af3_all_atom")
        proj_config = proj_entry.get_config_with_preset()
        config = proj_config.model

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)

        xl_noisy = torch.randn((batch_size, n_sample, n_atom, 3))
        t = torch.ones((batch_size, n_sample))
        atom_mask = torch.ones((batch_size, 1, n_atom))
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
            "ref_pos": torch.randn((batch_size, 1, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, 1, n_atom)),
            "ref_element": torch.ones((batch_size, 1, n_atom, 119)),
            "ref_charge": torch.ones((batch_size, 1, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, 1, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, 1, n_atom)),
            "num_atoms_per_token": torch.ones((batch_size, 1, n_token)) * 4,
        }

        xl = dm(
            batch=batch,
            xl_noisy=xl_noisy,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(xl.shape == (batch_size, n_sample, n_atom, 3))


class TestSampleDiffusion(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res

        proj_entry = registry.get_project_entry("af3_all_atom")
        proj_config = proj_entry.get_config_with_preset()
        config = proj_config.model

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z
        config.architecture.shared.no_mini_rollout_steps = 2
        config.architecture.shared.no_full_rollout_steps = 2

        sample_config = config.architecture.sample_diffusion

        dm = DiffusionModule(config=config.architecture.diffusion_module)
        sd = SampleDiffusion(**sample_config, diffusion_module=dm)

        batch = {
            "token_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 119)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
            "num_atoms_per_token": torch.ones((batch_size, n_token)) * 4,
        }

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        with torch.no_grad():
            noise_sched_config = config.architecture.noise_schedule
            noise_sched_config.no_rollout_steps = 2
            noise_schedule = create_noise_schedule(
                **noise_sched_config, dtype=si_input.dtype, device=si_input.device
            )

            xl = sd(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
            )

        self.assertTrue(xl.shape == (batch_size, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
