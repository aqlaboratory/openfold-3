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
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from tests.config import consts
from tests.data_utils import random_of3_features


class TestDiffusionModule(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()

        xl_noisy = torch.randn((batch_size, n_atom, 3))
        t = torch.ones(1)
        atom_mask = torch.ones((batch_size, n_atom))
        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

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
        n_sample = 3

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        xl_noisy = torch.randn((batch_size, n_sample, n_atom, 3))
        t = torch.ones((batch_size, n_sample))
        atom_mask = torch.ones((batch_size, 1, n_atom))
        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

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

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z
        no_rollout_samples = 5

        sample_config = config.architecture.sample_diffusion

        dm = DiffusionModule(config=config.architecture.diffusion_module)
        sd = SampleDiffusion(**sample_config, diffusion_module=dm)

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        with torch.no_grad():
            noise_sched_config = config.architecture.noise_schedule
            noise_schedule = create_noise_schedule(
                no_rollout_steps=2,
                **noise_sched_config,
                dtype=si_input.dtype,
                device=si_input.device,
            )

            xl = sd(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=no_rollout_samples,
            )

        self.assertTrue(xl.shape == (batch_size, no_rollout_samples, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
