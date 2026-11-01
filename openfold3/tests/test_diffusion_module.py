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

from openfold3.core.model.structure import diffusion_module
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.utils import compare_utils
from openfold3.tests.utils.data_utils import random_of3_features


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


def _sample_diffusion_inputs(device, n_token=consts.n_res, no_rollout_steps=4):
    """A randomly initialised SampleDiffusion with matching inputs on ``device``."""
    batch_size = 1

    proj_entry = OF3ProjectEntry()
    config = proj_entry.get_model_config_with_presets()
    c_s_input = config.architecture.shared.c_s_input
    c_s = config.architecture.shared.c_s
    c_z = config.architecture.shared.c_z

    dm = DiffusionModule(config=config.architecture.diffusion_module)
    sd = SampleDiffusion(**config.architecture.sample_diffusion, diffusion_module=dm)
    # Some output projections are zero-initialised, which would hide the
    # denoiser from the comparison; use small random weights everywhere.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in sd.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=generator))
    sd = sd.to(device).eval()

    batch = random_of3_features(
        batch_size=batch_size,
        n_token=n_token,
        n_msa=consts.n_seq,
        n_templ=consts.n_templ,
        is_eval=True,
    )
    batch = tensor_tree_map(lambda t: t.unsqueeze(1).to(device), batch)

    inputs = {
        "batch": batch,
        "si_input": torch.rand((batch_size, 1, n_token, c_s_input), device=device),
        "si_trunk": torch.rand((batch_size, 1, n_token, c_s), device=device),
        "zij_trunk": torch.rand((batch_size, 1, n_token, n_token, c_z), device=device),
        "noise_schedule": create_noise_schedule(
            no_rollout_steps=no_rollout_steps,
            **config.architecture.noise_schedule,
            dtype=torch.float32,
            device=device,
        ),
        "no_rollout_samples": 3,
        "use_conditioning": True,
    }
    return sd, inputs


def _rollout(sd, inputs, seed, **kwargs):
    torch.manual_seed(seed)
    with torch.no_grad():
        return sd(**inputs, **kwargs)


class TestSampleDiffusionCudaGraph(unittest.TestCase):
    def test_unsupported_reasons(self):
        x = torch.zeros(1)
        self.assertIn("cpu", diffusion_module._cuda_graph_unsupported_reason(x, False))
        with torch.enable_grad():
            if torch.cuda.is_available():
                x_cuda = x.cuda()
                self.assertIn(
                    "autograd",
                    diffusion_module._cuda_graph_unsupported_reason(x_cuda, False),
                )

    def test_cpu_falls_back_to_eager(self):
        """On CPU the flag is ignored and the rollout is the eager one."""
        sd, inputs = _sample_diffusion_inputs("cpu", no_rollout_steps=2)
        with mock.patch.object(diffusion_module, "_DiffusionStepGraph") as graph:
            out_graph = _rollout(sd, inputs, seed=0, use_cuda_graph=True)
        graph.assert_not_called()
        out_eager = _rollout(sd, inputs, seed=0)
        self.assertTrue(torch.equal(out_graph, out_eager))

    def test_cyclic_chains_fall_back_to_eager(self):
        sd, inputs = _sample_diffusion_inputs("cpu", no_rollout_steps=2)
        batch = inputs["batch"]
        self.assertIsNotNone(diffusion_module._cuda_graph_batch(batch))
        batch["cyclic_mask"] = torch.zeros_like(batch["token_mask"], dtype=torch.bool)
        graph_batch = diffusion_module._cuda_graph_batch(batch)
        self.assertNotIn("cyclic_mask", graph_batch)
        self.assertIn("token_atom_index", graph_batch)
        batch["cyclic_mask"][..., :4] = True
        self.assertIsNone(diffusion_module._cuda_graph_batch(batch))

    @compare_utils.skip_unless_gpu_available()
    def test_graph_matches_eager(self):
        """The graphed rollout gives the same coordinates as the eager rollout."""
        sd, inputs = _sample_diffusion_inputs("cuda", no_rollout_steps=8)
        out_eager = _rollout(sd, inputs, seed=0)
        with mock.patch.object(
            diffusion_module,
            "_DiffusionStepGraph",
            wraps=diffusion_module._DiffusionStepGraph,
        ) as graph:
            out_graph = _rollout(sd, inputs, seed=0, use_cuda_graph=True)
            # A new rollout captures again, and follows its own noise.
            out_graph_seed1 = _rollout(sd, inputs, seed=1, use_cuda_graph=True)
        self.assertEqual(graph.call_count, 2)
        self.assertTrue(torch.equal(out_graph, out_eager))
        self.assertTrue(torch.equal(out_graph_seed1, _rollout(sd, inputs, seed=1)))
        self.assertFalse(torch.equal(out_graph_seed1, out_graph))

    @compare_utils.skip_unless_gpu_available()
    def test_graph_matches_eager_with_chunk_tuning(self):
        """Chunk size tuning runs in the eager warm-up, outside the capture."""
        sd, inputs = _sample_diffusion_inputs("cuda", no_rollout_steps=4)
        tuner = sd.diffusion_module.diffusion_conditioning.chunk_size_tuner
        self.assertIsNotNone(tuner)
        out_graph = _rollout(sd, inputs, seed=0, use_cuda_graph=True, chunk_size=4)
        self.assertEqual(len(tuner.cached_chunk_sizes), 1)
        out_eager = _rollout(sd, inputs, seed=0, chunk_size=4)
        self.assertTrue(torch.equal(out_graph, out_eager))


if __name__ == "__main__":
    unittest.main()
