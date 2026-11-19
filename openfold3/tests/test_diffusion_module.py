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
from unittest.mock import patch

import torch

from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import random_of3_features
from openfold3.tests.utils.model_utils import initialize_model_weights


def _build_diffusion_module():
    proj_entry = OF3ProjectEntry()
    config = proj_entry.get_model_config_with_presets()
    dm = DiffusionModule(config=config.architecture.diffusion_module)
    initialize_model_weights(dm)
    dm.eval()
    return config, dm


class TestDiffusionModule(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        config, dm = _build_diffusion_module()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

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

        with torch.no_grad():
            xl = dm(
                batch=batch,
                xl_noisy=xl_noisy,
                t=t,
                token_mask=batch["token_mask"],
                atom_mask=atom_mask,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
            )

        self.assertTrue(xl.shape == (batch_size, n_atom, 3))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_sample = 3

        config, dm = _build_diffusion_module()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

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

        with torch.no_grad():
            xl = dm(
                batch=batch,
                xl_noisy=xl_noisy,
                token_mask=batch["token_mask"],
                atom_mask=atom_mask,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
            )

        self.assertTrue(xl.shape == (batch_size, n_sample, n_atom, 3))

    def test_cached_matches_uncached(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        config, dm = _build_diffusion_module()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

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
        step_kwargs = dict(
            batch=batch,
            xl_noisy=xl_noisy,
            t=t,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        for use_conditioning in (True, False):
            with self.subTest(use_conditioning=use_conditioning):
                with torch.no_grad():
                    cache = dm.prepare_cache(
                        batch=batch,
                        zij_trunk=zij_trunk,
                        use_conditioning=use_conditioning,
                    )
                    xl_cached = dm(
                        **step_kwargs,
                        use_conditioning=use_conditioning,
                        cache=cache,
                    )
                    xl_uncached = dm(
                        **step_kwargs,
                        use_conditioning=use_conditioning,
                        cache=None,
                    )
                self.assertTrue(torch.equal(xl_cached, xl_uncached))


class TestSampleDiffusion(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        config, dm = _build_diffusion_module()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z
        no_rollout_samples = 5

        sample_config = config.architecture.sample_diffusion
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
                use_conditioning=True,
            )

        self.assertTrue(xl.shape == (batch_size, no_rollout_samples, n_atom, 3))

    def test_prepare_cache_called_once_per_rollout(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        no_rollout_samples = 5
        no_rollout_steps = 2

        config, dm = _build_diffusion_module()
        sd = SampleDiffusion(
            **config.architecture.sample_diffusion, diffusion_module=dm
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z
        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))
        noise_schedule = create_noise_schedule(
            no_rollout_steps=no_rollout_steps,
            **config.architecture.noise_schedule,
            dtype=si_input.dtype,
            device=si_input.device,
        )

        with (
            patch.object(dm, "prepare_cache", wraps=dm.prepare_cache) as mock_cache,
            patch.object(
                dm.diffusion_conditioning,
                "condition_pair",
                wraps=dm.diffusion_conditioning.condition_pair,
            ) as mock_pair,
            torch.no_grad(),
        ):
            sd(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=no_rollout_samples,
                use_conditioning=True,
            )

        self.assertEqual(mock_cache.call_count, 1)
        self.assertEqual(mock_pair.call_count, 1)


if __name__ == "__main__":
    unittest.main()
