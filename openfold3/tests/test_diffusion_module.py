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

from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import random_of3_features, randomize_parameters


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
            use_conditioning=True,
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
            use_conditioning=True,
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
                use_conditioning=True,
            )

        self.assertTrue(xl.shape == (batch_size, no_rollout_samples, n_atom, 3))


class TestStepInvariants(unittest.TestCase):
    """
    `precompute_step_invariants` computes the quantities that a diffusion rollout
    recomputes at every step even though they do not change (the conditioned pair
    representation, the atom reference/trunk embeddings, and the attention mask
    biases). Using them via `step_invariants` / `hoist_step_invariants` must give
    the same result as recomputing everything at every call.
    """

    def _setup(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = randomize_parameters(
            DiffusionModule(config=config.architecture.diffusion_module).eval(),
            std=0.05,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        # Padded atoms, so that the attention mask biases are not trivially zero
        batch["atom_mask"][..., -5:] = 0

        inputs = {
            "si_input": torch.randn((batch_size, 1, n_token, c_s_input)),
            "si_trunk": torch.randn((batch_size, 1, n_token, c_s)),
            "zij_trunk": torch.randn((batch_size, 1, n_token, n_token, c_z)),
        }

        return config, dm, batch, n_atom, inputs

    def test_diffusion_module(self):
        # The diffusion module recomputes the same quantities at every step of a rollout, so precomputing them must give the same result as recomputing them at every step.
        n_sample = 3
        config, dm, batch, n_atom, inputs = self._setup()
        batch_size = consts.batch_size

        xl_noisy = torch.randn((batch_size, n_sample, n_atom, 3))
        atom_mask = batch["atom_mask"]

        for use_conditioning in (True, False):
            with self.subTest(use_conditioning=use_conditioning), torch.no_grad():
                step_invariants = dm.precompute_step_invariants(
                    batch=batch,
                    si_trunk=inputs["si_trunk"],
                    zij_trunk=inputs["zij_trunk"],
                    use_conditioning=use_conditioning,
                )

                # The same precomputed values are used for every noise level
                for t in (torch.ones(()), torch.full((), 0.3)):
                    kwargs = {
                        "batch": batch,
                        "xl_noisy": xl_noisy,
                        "token_mask": batch["token_mask"],
                        "atom_mask": atom_mask,
                        "t": t,
                        "use_conditioning": use_conditioning,
                        **inputs,
                    }
                    expected = dm(**kwargs)
                    actual = dm(**kwargs, step_invariants=step_invariants)

                    self.assertTrue(torch.isfinite(expected).all())
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_sample_diffusion(self):
        # The diffusion rollout recomputes the same quantities at every step, so precomputing them must give the same result as recomputing them at every step.
        no_rollout_samples = 2
        config, dm, batch, n_atom, inputs = self._setup()

        sd = SampleDiffusion(
            **config.architecture.sample_diffusion, diffusion_module=dm
        )

        with torch.no_grad():
            noise_schedule = create_noise_schedule(
                no_rollout_steps=3,
                **config.architecture.noise_schedule,
                dtype=inputs["si_input"].dtype,
                device=inputs["si_input"].device,
            )

            results = {}
            for hoist in (False, True):
                # Precomputing must not consume any random numbers
                torch.manual_seed(1234)
                results[hoist] = sd(
                    batch=batch,
                    noise_schedule=noise_schedule,
                    no_rollout_samples=no_rollout_samples,
                    use_conditioning=True,
                    hoist_step_invariants=hoist,
                    **inputs,
                )

        self.assertTrue(torch.isfinite(results[False]).all())
        torch.testing.assert_close(results[True], results[False], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
