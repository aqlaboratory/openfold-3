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
    DiffusionConditioning,
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)
from tests.config import consts


class TestDiffusionConditioning(unittest.TestCase):
    def test_diffusion_conditioning_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        c_fourier_emb = 256
        max_relative_idx = 32
        max_relative_chain = 2
        sigma_data = 16

        dc = DiffusionConditioning(
            c_s_input=c_s_input,
            c_s=c_s,
            c_z=c_z,
            c_fourier_emb=c_fourier_emb,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            sigma_data=sigma_data,
        )

        t = torch.ones(batch_size)
        s_input = torch.rand((batch_size, n_token, c_s_input))
        s_trunk = torch.rand((batch_size, n_token, c_s))
        z_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        pair_token_mask = torch.ones((batch_size, n_token, n_token))

        batch = {
            "token_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "residue_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

        out_s, out_z = dc(
            batch=batch,
            t=t,
            s_input=s_input,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            token_mask=token_mask,
            pair_token_mask=pair_token_mask,
        )

        self.assertTrue(out_s.shape == (batch_size, n_token, c_s))
        self.assertTrue(out_z.shape == (batch_size, n_token, n_token, c_z))


class TestDiffusionModule(unittest.TestCase):
    def test_diffusion_module_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        c_fourier_emb = 256
        max_relative_idx = 32
        max_relative_chain = 2
        sigma_data = 16
        c_atom_ref = 390
        c_atom = 64
        c_atom_pair = 16
        c_token = 768
        c_hidden_att = 48  # c_token / no_heads
        no_heads = 16
        no_blocks = 24
        n_transition = 2
        inf = 1e5

        dm = DiffusionModule(
            c_s_input=c_s_input,
            c_s=c_s,
            c_z=c_z,
            c_fourier_emb=c_fourier_emb,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            c_atom_ref=c_atom_ref,
            sigma_data=sigma_data,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden_att=c_hidden_att,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            inf=inf,
        )

        x_noisy = torch.randn((batch_size, n_atom, 3))
        t = torch.ones(1)
        s_input = torch.rand((batch_size, n_token, c_s_input))
        s_trunk = torch.rand((batch_size, n_token, c_s))
        z_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        pair_token_mask = torch.ones((batch_size, n_token, n_token))
        atom_mask = torch.ones((batch_size, n_atom))

        batch = {
            "token_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "residue_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
        }

        out = dm(
            batch=batch,
            x_noisy=x_noisy,
            t=t,
            s_input=s_input,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            token_mask=token_mask,
            pair_token_mask=pair_token_mask,
            atom_mask=atom_mask,
        )

        self.assertTrue(out.shape == (batch_size, n_atom, 3))


class TestSampleDiffusion(unittest.TestCase):
    def test_sample_diffusion_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        c_fourier_emb = 256
        max_relative_idx = 32
        max_relative_chain = 2
        sigma_data = 16
        c_atom_ref = 390
        c_atom = 64
        c_atom_pair = 16
        c_token = 768
        c_hidden_att = 48  # c_token / no_heads
        no_heads = 16
        no_blocks = 24
        n_transition = 2
        inf = 1e5
        gamma_0 = 0.8
        gamma_min = 1.0
        noise_scale = 1.003
        step_scale = 1.5
        s_max = 160
        s_min = 4e-4
        p = 7
        T = 5
        step_size = 1.0 / T

        sd = SampleDiffusion(
            c_s_input=c_s_input,
            c_s=c_s,
            c_z=c_z,
            c_fourier_emb=c_fourier_emb,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            c_atom_ref=c_atom_ref,
            sigma_data=sigma_data,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden_att=c_hidden_att,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            inf=inf,
            gamma_0=gamma_0,
            gamma_min=gamma_min,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )

        s_input = torch.rand((batch_size, n_token, c_s_input))
        s_trunk = torch.rand((batch_size, n_token, c_s))
        z_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        pair_token_mask = torch.ones((batch_size, n_token, n_token))
        atom_mask = torch.ones((batch_size, n_atom))

        batch = {
            "token_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "residue_index": torch.arange(0, n_token)
            .unsqueeze(0)
            .repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
        }

        noise_schedule = create_noise_schedule(
            step_size=step_size, sigma_data=sigma_data, s_max=s_max, s_min=s_min, p=p
        )

        with torch.no_grad():
            out = sd(
                batch=batch,
                s_input=s_input,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                token_mask=token_mask,
                pair_token_mask=pair_token_mask,
                atom_mask=atom_mask,
                noise_schedule=noise_schedule,
            )

        self.assertTrue(noise_schedule.shape == (T + 1,))
        self.assertTrue(out.shape == (batch_size, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
