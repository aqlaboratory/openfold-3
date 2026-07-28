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
from openfold3.core.utils.relpos import relpos_complex
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts


class TestDiffusionConditioning(unittest.TestCase):
    def _batch(self, batch_size: int, n_token: int) -> dict:
        return {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

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

        batch = self._batch(batch_size, n_token)

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

    def test_relpos_row_slice_matches_full(self):
        batch_size = 2
        n_token = 17
        batch = self._batch(batch_size, n_token)
        full = relpos_complex(batch, max_relative_idx=32, max_relative_chain=2)
        chunk = 5
        parts = [
            relpos_complex(
                batch,
                max_relative_idx=32,
                max_relative_chain=2,
                row_slice=slice(i, min(i + chunk, n_token)),
            )
            for i in range(0, n_token, chunk)
        ]
        rebuilt = torch.cat(parts, dim=-3)
        self.assertTrue(torch.equal(rebuilt, full))

    def test_chunked_pair_embed_matches_eager(self):
        batch_size = 1
        n_token = 37
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})
        dc = DiffusionConditioning(**diff_cond_config)
        dc.eval()

        torch.manual_seed(0)
        batch = self._batch(batch_size, n_token)
        zij_trunk = torch.randn(batch_size, n_token, n_token, c_z)

        with torch.no_grad():
            zij_chunked = dc._embed_zij_chunked(batch, zij_trunk)
        # Eager path is taken when gradients are enabled.
        with torch.enable_grad():
            zij_eager = dc._embed_zij(batch, zij_trunk.detach())

        self.assertTrue(
            torch.allclose(zij_chunked, zij_eager, atol=1e-5, rtol=1e-5),
            f"max abs diff={(zij_chunked - zij_eager).abs().max().item():.3e}",
        )

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


if __name__ == "__main__":
    unittest.main()
