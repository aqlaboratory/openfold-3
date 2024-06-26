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

import torch
import unittest

from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
)
from openfold3.model_implementations.af3_all_atom.config import config
from tests.config import consts


class TestDiffusionModule(unittest.TestCase):

    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res

        c_s_input = config.globals.c_s_input
        c_s = config.globals.c_s
        c_z = config.globals.c_z

        dm = DiffusionModule(config=config.model.diffusion_module)

        xl_noisy = torch.randn((batch_size, n_atom, 3))
        t = torch.ones(1)
        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        batch = {
            'token_index': torch.arange(0, n_token).unsqueeze(0).repeat((batch_size, 1)),
            'token_mask': torch.ones((batch_size, n_token)),
            'residue_index': torch.arange(0, n_token).unsqueeze(0).repeat((batch_size, 1)),
            'sym_id': torch.zeros((batch_size, n_token)),
            'asym_id': torch.zeros((batch_size, n_token)),
            'entity_id': torch.zeros((batch_size, n_token)),
            'ref_pos': torch.randn((batch_size, n_atom, 3)),
            'ref_mask': torch.ones((batch_size, n_atom)),
            'ref_element': torch.ones((batch_size, n_atom, 128)),
            'ref_charge': torch.ones((batch_size, n_atom)),
            'ref_atom_name_chars': torch.ones((batch_size, n_atom, 4, 64)),
            'ref_space_uid': torch.zeros((batch_size, n_atom)),
            'atom_to_token_index': torch.eye(n_token).repeat_interleave(4, dim=0).unsqueeze(0).repeat(batch_size, 1, 1),
        }

        xl = dm(batch=batch,
                xl_noisy=xl_noisy,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk)
        
        self.assertTrue(xl.shape == (batch_size, n_atom, 3))
    
    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        n_sample = 3

        c_s_input = config.globals.c_s_input
        c_s = config.globals.c_s
        c_z = config.globals.c_z

        dm = DiffusionModule(config=config.model.diffusion_module)

        xl_noisy = torch.randn((batch_size, n_sample, n_atom, 3))
        t = torch.ones((1, 1))
        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))

        batch = {
            'token_index': torch.arange(0, n_token)[None, None, :].repeat((batch_size, 1, 1)),
            'token_mask': torch.ones((batch_size, 1, n_token)),
            'residue_index': torch.arange(0, n_token)[None, None, :].repeat((batch_size, 1, 1)),
            'sym_id': torch.zeros((batch_size, 1, n_token)),
            'asym_id': torch.zeros((batch_size, 1, n_token)),
            'entity_id': torch.zeros((batch_size, 1, n_token)),
            'ref_pos': torch.randn((batch_size, 1, n_atom, 3)),
            'ref_mask': torch.ones((batch_size, 1, n_atom)),
            'ref_element': torch.ones((batch_size, 1, n_atom, 128)),
            'ref_charge': torch.ones((batch_size, 1, n_atom)),
            'ref_atom_name_chars': torch.ones((batch_size, 1, n_atom, 4, 64)),
            'ref_space_uid': torch.zeros((batch_size, 1, n_atom)),
            'atom_to_token_index': torch.eye(n_token).repeat_interleave(4, dim=0)[None, None, :, :].repeat(batch_size, 1, 1, 1),
        }

        xl = dm(batch=batch,
                xl_noisy=xl_noisy,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk)
        
        self.assertTrue(xl.shape == (batch_size, n_sample, n_atom, 3))


class TestSampleDiffusion(unittest.TestCase):

    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res

        c_s_input = config.globals.c_s_input
        c_s = config.globals.c_s
        c_z = config.globals.c_z

        sample_config = config.model.sample_diffusion
        sample_config.no_rollout_steps = 2

        dm = DiffusionModule(config=config.model.diffusion_module)
        sd = SampleDiffusion(**sample_config, diffusion_module=dm)

        batch = {
            'token_index': torch.arange(0, n_token).unsqueeze(0).repeat((batch_size, 1)),
            'token_mask': torch.ones((batch_size, n_token)),
            'residue_index': torch.arange(0, n_token).unsqueeze(0).repeat((batch_size, 1)),
            'sym_id': torch.zeros((batch_size, n_token)),
            'asym_id': torch.zeros((batch_size, n_token)),
            'entity_id': torch.zeros((batch_size, n_token)),
            'ref_pos': torch.randn((batch_size, n_atom, 3)),
            'ref_mask': torch.ones((batch_size, n_atom)),
            'ref_element': torch.ones((batch_size, n_atom, 128)),
            'ref_charge': torch.ones((batch_size, n_atom)),
            'ref_atom_name_chars': torch.ones((batch_size, n_atom, 4, 64)),
            'ref_space_uid': torch.zeros((batch_size, n_atom)),
            'atom_to_token_index': torch.eye(n_token).repeat_interleave(4, dim=0).unsqueeze(0).repeat(batch_size, 1, 1),
        }

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))

        with torch.no_grad():
            xl = sd(batch=batch,
                    si_input=si_input,
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk)

        self.assertTrue(xl.shape == (batch_size, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
