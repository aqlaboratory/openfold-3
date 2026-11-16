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


def _build_diffusion_conditioning():
    proj_entry = OF3ProjectEntry()
    config = proj_entry.get_model_config_with_presets()
    diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
    diff_cond_config.update(
        {
            "c_s": consts.c_s,
            "c_s_input": consts.c_s + 65,
            "c_z": consts.c_z,
        }
    )
    return DiffusionConditioning(**diff_cond_config), diff_cond_config


def _relpos_batch(*, sample_channel: bool) -> dict[str, torch.Tensor]:
    batch_size = consts.batch_size
    n_token = consts.n_res
    token_shape: tuple[int, ...]
    if sample_channel:
        idx = torch.arange(0, n_token)[None, None, :].repeat((batch_size, 1, 1))
        token_shape = (batch_size, 1, n_token)
    else:
        idx = torch.arange(0, n_token)[None, :].repeat((batch_size, 1))
        token_shape = (batch_size, n_token)
    return {
        "token_index": idx,
        "token_mask": torch.ones(token_shape),
        "residue_index": idx.clone(),
        "sym_id": torch.zeros(token_shape),
        "asym_id": torch.zeros(token_shape),
        "entity_id": torch.zeros(token_shape),
        "cyclic_mask": torch.zeros((batch_size, 1, n_token), dtype=torch.bool),
    }


def _trunk_feats(
    *, sample_channel: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = consts.batch_size
    n_token = consts.n_res
    c_s_input = consts.c_s + 65
    c_s = consts.c_s
    c_z = consts.c_z
    token_shape: tuple[int, ...]
    pair_shape: tuple[int, ...]
    if sample_channel:
        token_shape = (batch_size, 1, n_token)
        pair_shape = (batch_size, 1, n_token, n_token, c_z)
    else:
        token_shape = (batch_size, n_token)
        pair_shape = (batch_size, n_token, n_token, c_z)
    return (
        torch.rand((*token_shape, c_s_input)),
        torch.rand((*token_shape, c_s)),
        torch.rand(pair_shape),
    )


class TestDiffusionConditioning(unittest.TestCase):
    def setUp(self):
        self.dc, self.diff_cond_config = _build_diffusion_conditioning()
        self.batch_size = consts.batch_size
        self.n_token = consts.n_res
        self.c_s = consts.c_s
        self.c_z = consts.c_z

    def test_without_n_sample_channel(self):
        si_input, si_trunk, zij_trunk = _trunk_feats(sample_channel=False)
        t = self.diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn(self.batch_size, device=si_trunk.device)
        )
        batch = _relpos_batch(sample_channel=False)

        si, zij = self.dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (self.batch_size, self.n_token, self.c_s))
        self.assertTrue(
            zij.shape == (self.batch_size, self.n_token, self.n_token, self.c_z)
        )

    def test_with_different_schedule(self):
        n_sample = 3
        si_input, si_trunk, zij_trunk = _trunk_feats(sample_channel=True)
        t = self.diff_cond_config.sigma_data * torch.exp(
            -1.2
            + 1.5 * torch.randn((self.batch_size, n_sample), device=si_trunk.device)
        )
        batch = _relpos_batch(sample_channel=True)

        si, zij = self.dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (self.batch_size, n_sample, self.n_token, self.c_s))
        self.assertTrue(
            zij.shape == (self.batch_size, 1, self.n_token, self.n_token, self.c_z)
        )

    def test_with_same_schedule(self):
        si_input, si_trunk, zij_trunk = _trunk_feats(sample_channel=True)
        t = self.diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn((1, 1), device=si_trunk.device)
        )
        batch = _relpos_batch(sample_channel=True)

        si, zij = self.dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )

        self.assertTrue(si.shape == (self.batch_size, 1, self.n_token, self.c_s))
        self.assertTrue(
            zij.shape == (self.batch_size, 1, self.n_token, self.n_token, self.c_z)
        )

    def test_condition_pair_matches_forward(self):
        self.dc.eval()
        si_input, si_trunk, zij_trunk = _trunk_feats(sample_channel=False)
        t = self.diff_cond_config.sigma_data * torch.exp(
            -1.2 + 1.5 * torch.randn(self.batch_size, device=si_trunk.device)
        )
        batch = _relpos_batch(sample_channel=False)

        kwargs = dict(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            use_conditioning=True,
        )
        with torch.no_grad():
            si_fwd, zij_fwd = self.dc(t=t, zij_trunk=zij_trunk, **kwargs)
            zij = self.dc.condition_pair(
                batch=batch, zij_trunk=zij_trunk, use_conditioning=True
            )
            si_cached = self.dc.condition_single(t=t, **kwargs)

        self.assertTrue(torch.equal(si_fwd, si_cached))
        self.assertTrue(torch.equal(zij_fwd, zij))

        t2 = t * 1.1
        with torch.no_grad():
            si_t2 = self.dc.condition_single(t=t2, **kwargs)
        self.assertFalse(torch.equal(si_cached, si_t2))


if __name__ == "__main__":
    unittest.main()
