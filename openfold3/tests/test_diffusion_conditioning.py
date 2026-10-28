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
from unittest import mock

import torch

from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts


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


class TestEmbedZijChunking(unittest.TestCase):
    """The row-chunked pair conditioning must match the eager one."""

    N_TOKEN = consts.n_res
    # Two chains, the second cyclic, so row slices cross a chain boundary and
    # land inside a cyclic chain that has to build its offsets the long way.
    ASYM_IDS = [1] * 11 + [2] * 11
    CYCLIC_MASK = [False] * 11 + [True] * 11

    def _make_dc(self, tune_chunk_size):
        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
        diff_cond_config.update(
            {"c_s": consts.c_s, "c_s_input": consts.c_s + 65, "c_z": consts.c_z}
        )
        return DiffusionConditioning(
            **{**diff_cond_config, "tune_chunk_size": tune_chunk_size}
        )

    def _make_batch(self):
        batch_size = consts.batch_size
        n_token = self.N_TOKEN
        return {
            "token_index": torch.arange(n_token)[None, :].repeat(batch_size, 1),
            "residue_index": torch.arange(n_token)[None, :].repeat(batch_size, 1),
            "sym_id": torch.ones(batch_size, n_token, dtype=torch.int32),
            "asym_id": torch.tensor(self.ASYM_IDS, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(batch_size, 1),
            "entity_id": torch.tensor(self.ASYM_IDS, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(batch_size, 1),
            "cyclic_mask": torch.tensor(self.CYCLIC_MASK, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(batch_size, 1),
        }

    def test_chunked_matches_eager(self):
        dc = self._make_dc(tune_chunk_size=False)
        dc.eval()
        batch = self._make_batch()
        zij_trunk = torch.rand(
            consts.batch_size, self.N_TOKEN, self.N_TOKEN, consts.c_z
        )

        # The chunked path is gated on inference, so no_grad is required for it
        # to be taken at all.
        with torch.no_grad():
            expected = dc._embed_zij(batch=batch, zij_trunk=zij_trunk)

            # Both chunk sizes divide N_TOKEN unevenly, so the last chunk of
            # each is short.
            for chunk_size in (3, 5, 7):
                assert self.N_TOKEN % chunk_size != 0, chunk_size
                with self.subTest(chunk_size=chunk_size):
                    actual = dc._embed_zij(
                        batch=batch, zij_trunk=zij_trunk, chunk_size=chunk_size
                    )
                    torch.testing.assert_close(actual, expected)

    def test_chunked_is_skipped_under_grad(self):
        # The eager path is the fallback: training needs the graph, and the
        # chunked path writes into a preallocated buffer that would not carry it.
        dc = self._make_dc(tune_chunk_size=False)
        batch = self._make_batch()
        zij_trunk = torch.rand(
            consts.batch_size, self.N_TOKEN, self.N_TOKEN, consts.c_z
        )

        # Counting calls is the only reliable observable here: an index write
        # into torch.empty still yields requires_grad=True, so that flag cannot
        # distinguish the two branches.
        with mock.patch.object(
            dc, "_embed_zij_chunked", wraps=dc._embed_zij_chunked
        ) as chunked:
            dc._embed_zij(batch=batch, zij_trunk=zij_trunk, chunk_size=5)
        assert chunked.call_count == 0

        # And it does run at inference, so the zero above is the gate talking.
        dc.eval()
        with (
            torch.no_grad(),
            mock.patch.object(
                dc, "_embed_zij_chunked", wraps=dc._embed_zij_chunked
            ) as chunked,
        ):
            dc._embed_zij(batch=batch, zij_trunk=zij_trunk, chunk_size=5)
        assert chunked.call_count == 1

    def test_chunk_size_tuner_is_used_and_keeps_the_result(self):
        dc = self._make_dc(tune_chunk_size=True)
        dc.eval()
        batch = self._make_batch()
        zij_trunk = torch.rand(
            consts.batch_size, self.N_TOKEN, self.N_TOKEN, consts.c_z
        )

        with torch.no_grad():
            expected = dc._embed_zij(batch=batch, zij_trunk=zij_trunk)
            actual = dc._embed_zij(batch=batch, zij_trunk=zij_trunk, chunk_size=8)

        torch.testing.assert_close(actual, expected)
        assert len(dc.chunk_size_tuner.cached_chunk_sizes) == 1
        _, max_chunk_size, _ = dc.chunk_size_tuner.cached_chunk_sizes[0]
        assert max_chunk_size == 8


if __name__ == "__main__":
    unittest.main()
