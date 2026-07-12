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

"""Numerical equivalence between the unrolled DiffusionConditioning forward
and the prepare_invariants + apply_t pair used inside the rollout loop.

The cache split is a math-preserving refactor (the loop-invariant ops are
just hoisted out of the per-step path), so we expect bitwise-identical
outputs at fp32. ``apply_t`` returns ``cache["zij_conditioned"]`` by
reference; a separate test verifies the cache is not mutated across
iterations.
"""

import unittest

import torch

from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionEncoder,
    NoisyPositionEmbedder,
)
from openfold3.core.model.structure.diffusion_module import DiffusionModule
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.data_utils import random_of3_features


def _build_dc(c_s_input: int, c_s: int, c_z: int) -> DiffusionConditioning:
    proj_entry = OF3ProjectEntry()
    config = proj_entry.get_model_config_with_presets()
    diff_cond_config = config.architecture.diffusion_module.diffusion_conditioning
    diff_cond_config.update({"c_s": c_s, "c_s_input": c_s_input, "c_z": c_z})
    dc = DiffusionConditioning(**diff_cond_config)
    dc.eval()
    return dc, diff_cond_config


def _sample_t(diff_cond_config, shape):
    return diff_cond_config.sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(*shape))


class TestDiffusionConditioningCache(unittest.TestCase):
    def test_prepare_apply_numerical_equivalence_no_sample_dim(self):
        """forward(t) ≡ apply_t(prepare_invariants(...), t) over multiple t."""
        torch.manual_seed(0)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        dc, diff_cond_config = _build_dc(c_s_input, c_s, c_z)

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        batch = {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": token_mask,
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

        with torch.no_grad():
            cache = dc.prepare_invariants(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
                token_mask=token_mask,
            )
            for _ in range(5):
                t = _sample_t(diff_cond_config, (batch_size,))
                si_ref, zij_ref = dc(
                    batch=batch,
                    t=t,
                    si_input=si_input,
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk,
                    use_conditioning=True,
                )
                si_cached, zij_cached = dc.apply_t(
                    cache=cache, t=t, token_mask=token_mask
                )
                torch.testing.assert_close(si_cached, si_ref, rtol=0.0, atol=0.0)
                torch.testing.assert_close(zij_cached, zij_ref, rtol=0.0, atol=0.0)

    def test_prepare_apply_numerical_equivalence_with_sample_dim(self):
        """[B, n_sample, N, ...] layout from SampleDiffusion."""
        torch.manual_seed(1)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z
        n_sample = 3

        dc, diff_cond_config = _build_dc(c_s_input, c_s, c_z)

        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, 1, n_token))
        batch = {
            "token_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "token_mask": token_mask,
            "residue_index": torch.arange(0, n_token)[None, None, :].repeat(
                (batch_size, 1, 1)
            ),
            "sym_id": torch.zeros((batch_size, 1, n_token)),
            "asym_id": torch.zeros((batch_size, 1, n_token)),
            "entity_id": torch.zeros((batch_size, 1, n_token)),
        }

        with torch.no_grad():
            cache = dc.prepare_invariants(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
                token_mask=token_mask,
            )
            for _ in range(3):
                t = _sample_t(diff_cond_config, (batch_size, n_sample))
                si_ref, zij_ref = dc(
                    batch=batch,
                    t=t,
                    si_input=si_input,
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk,
                    use_conditioning=True,
                )
                si_cached, zij_cached = dc.apply_t(
                    cache=cache, t=t, token_mask=token_mask
                )
                torch.testing.assert_close(si_cached, si_ref, rtol=0.0, atol=0.0)
                torch.testing.assert_close(zij_cached, zij_ref, rtol=0.0, atol=0.0)

    def test_prepare_apply_use_conditioning_false(self):
        """Zero-out fires once in prepare; per-step apply_t still matches."""
        torch.manual_seed(2)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        dc, diff_cond_config = _build_dc(c_s_input, c_s, c_z)

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        batch = {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": token_mask,
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }
        t = _sample_t(diff_cond_config, (batch_size,))

        with torch.no_grad():
            si_ref, zij_ref = dc(
                batch=batch,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=False,
            )
            cache = dc.prepare_invariants(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=False,
                token_mask=token_mask,
            )
            si_cached, zij_cached = dc.apply_t(
                cache=cache, t=t, token_mask=token_mask
            )
            torch.testing.assert_close(si_cached, si_ref, rtol=0.0, atol=0.0)
            torch.testing.assert_close(zij_cached, zij_ref, rtol=0.0, atol=0.0)

    def test_zij_cached_unchanged_across_iterations(self):
        """apply_t never mutates the cached zij tensor."""
        torch.manual_seed(3)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        dc, diff_cond_config = _build_dc(c_s_input, c_s, c_z)

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        token_mask = torch.ones((batch_size, n_token))
        batch = {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": token_mask,
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

        with torch.no_grad():
            cache = dc.prepare_invariants(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
                token_mask=token_mask,
            )
            zij_initial = cache["zij_conditioned"].clone()
            for _ in range(3):
                t = _sample_t(diff_cond_config, (batch_size,))
                dc.apply_t(cache=cache, t=t, token_mask=token_mask)
            self.assertTrue(torch.equal(cache["zij_conditioned"], zij_initial))

    def test_forward_training_path_unchanged(self):
        """forward without a cache argument still works (training path)."""
        torch.manual_seed(4)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s_input = consts.c_s + 65
        c_s = consts.c_s
        c_z = consts.c_z

        dc, diff_cond_config = _build_dc(c_s_input, c_s, c_z)
        dc.train()

        si_input = torch.rand((batch_size, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, n_token, c_s))
        zij_trunk = torch.rand((batch_size, n_token, n_token, c_z))
        t = _sample_t(diff_cond_config, (batch_size,))
        batch = {
            "token_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "token_mask": torch.ones((batch_size, n_token)),
            "residue_index": torch.arange(0, n_token)[None, :].repeat((batch_size, 1)),
            "sym_id": torch.zeros((batch_size, n_token)),
            "asym_id": torch.zeros((batch_size, n_token)),
            "entity_id": torch.zeros((batch_size, n_token)),
        }

        si, zij = dc(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
        )
        self.assertEqual(si.shape, (batch_size, n_token, c_s))
        self.assertEqual(zij.shape, (batch_size, n_token, n_token, c_z))


class TestDiffusionModuleCacheKwarg(unittest.TestCase):
    def test_cache_kwarg_matches_uncached(self):
        """DiffusionModule.forward with and without conditioning_cache match."""
        torch.manual_seed(5)
        batch_size = consts.batch_size
        n_token = consts.n_res

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)
        dm.eval()

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        # Add a sample dim like SampleDiffusion does (B -> B, 1) for trunk inputs.
        batch = tensor_tree_map(lambda x: x.unsqueeze(1), batch)
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()

        xl_noisy = torch.randn((batch_size, 1, n_atom, 3))
        t = torch.ones(batch_size, 1)
        atom_mask = torch.ones((batch_size, 1, n_atom))
        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))
        token_mask = batch["token_mask"]

        with torch.no_grad():
            xl_uncached = dm(
                batch=batch,
                xl_noisy=xl_noisy,
                token_mask=token_mask,
                atom_mask=atom_mask,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
            )
            cache = dm.prepare_diffusion_conditioning_cache(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
            )
            xl_cached = dm(
                batch=batch,
                xl_noisy=xl_noisy,
                token_mask=token_mask,
                atom_mask=atom_mask,
                t=t,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=True,
                conditioning_cache=cache,
            )
            # Same conditioning math, identical xl_noisy/t — outputs must be
            # bitwise identical aside from any nondeterministic atom-attention
            # kernels. The eval-time path here is deterministic at fp32.
            torch.testing.assert_close(xl_cached, xl_uncached, rtol=0.0, atol=0.0)


class TestNoisyPositionEmbedderSplit(unittest.TestCase):
    def test_embed_trunk_plus_embed_rl_matches_forward(self):
        """``embed_trunk(...) + embed_rl(...)`` must equal ``forward(...)``."""
        torch.manual_seed(6)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 128
        c_atom_pair = 16
        n_query = 32
        n_key = 128

        embedder = NoisyPositionEmbedder(
            c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair,
        )
        embedder.eval()

        batch = random_of3_features(
            batch_size=batch_size, n_token=n_token,
            n_msa=consts.n_seq, n_templ=consts.n_templ,
        )
        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = -(-n_atom // n_query)

        cl = torch.randn((batch_size, n_atom, c_atom))
        plm = torch.randn((batch_size, num_blocks, n_query, n_key, c_atom_pair))
        si_trunk = torch.randn((batch_size, n_token, c_s))
        zij_trunk = torch.randn((batch_size, n_token, n_token, c_z))
        rl = torch.randn((batch_size, n_atom, 3))

        with torch.no_grad():
            cl_ref, plm_ref, ql_ref = embedder(
                batch=batch, cl=cl, plm=plm,
                si_trunk=si_trunk, zij_trunk=zij_trunk, rl=rl,
                n_query=n_query, n_key=n_key,
            )
            cl_split, plm_split = embedder.embed_trunk(
                batch=batch, cl=cl, plm=plm,
                si_trunk=si_trunk, zij_trunk=zij_trunk,
                n_query=n_query, n_key=n_key,
            )
            ql_split = embedder.embed_rl(cl=cl_split, rl=rl)

        torch.testing.assert_close(cl_split, cl_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(plm_split, plm_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(ql_split, ql_ref, rtol=0.0, atol=0.0)


class TestAtomAttentionEncoderCache(unittest.TestCase):
    def test_atom_rep_cache_matches_uncached(self):
        """forward(..., atom_rep_cache=cache) ≡ forward(...) for fixed inputs."""
        torch.manual_seed(7)
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 128
        c_atom_pair = 16
        c_token = 384
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        c_hidden = c_atom // no_heads
        n_query = 32
        n_key = 128
        n_sample = 3

        # Local import to keep the per-test setup contained.
        from openfold3.tests.test_sequence_local_atom_attention import C_ATOM_REF

        atom_attn_enc = AtomAttentionEncoder(
            c_s=c_s, c_z=c_z, c_atom_ref=C_ATOM_REF, c_atom=c_atom,
            c_atom_pair=c_atom_pair, c_token=c_token, c_hidden=c_hidden,
            add_noisy_pos=True, no_heads=no_heads, no_blocks=no_blocks,
            n_transition=n_transition, n_query=n_query, n_key=n_key,
            use_ada_layer_norm=True,
        )
        atom_attn_enc.eval()

        batch = random_of3_features(
            batch_size=batch_size, n_token=n_token,
            n_msa=consts.n_seq, n_templ=consts.n_templ,
        )
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        n_atom = batch["ref_pos"].shape[-2]

        si_trunk = torch.randn((batch_size, 1, n_token, c_s))
        zij_trunk = torch.randn((batch_size, 1, n_token, n_token, c_z))

        with torch.no_grad():
            cache = atom_attn_enc.prepare_atom_rep_cache(
                batch=batch, si_trunk=si_trunk, zij_trunk=zij_trunk,
            )
            for _ in range(3):
                rl = torch.randn((batch_size, n_sample, n_atom, 3))
                ai_ref, ql_ref, cl_ref, plm_ref = atom_attn_enc(
                    batch=batch, rl=rl, si_trunk=si_trunk, zij_trunk=zij_trunk,
                )
                ai_c, ql_c, cl_c, plm_c = atom_attn_enc(
                    batch=batch, rl=rl, si_trunk=si_trunk, zij_trunk=zij_trunk,
                    atom_rep_cache=cache,
                )
                torch.testing.assert_close(ai_c, ai_ref, rtol=0.0, atol=0.0)
                torch.testing.assert_close(ql_c, ql_ref, rtol=0.0, atol=0.0)
                torch.testing.assert_close(cl_c, cl_ref, rtol=0.0, atol=0.0)
                torch.testing.assert_close(plm_c, plm_ref, rtol=0.0, atol=0.0)


class TestDiffusionModuleAtomRepCache(unittest.TestCase):
    def test_atom_rep_cache_matches_uncached(self):
        """DiffusionModule.forward(..., atom_rep_cache=...) matches the uncached
        path when both are given the same conditioning_cache (and rl/t)."""
        torch.manual_seed(8)
        batch_size = consts.batch_size
        n_token = consts.n_res

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        c_s_input = config.architecture.shared.c_s_input
        c_s = config.architecture.shared.c_s
        c_z = config.architecture.shared.c_z

        dm = DiffusionModule(config=config.architecture.diffusion_module)
        dm.eval()

        batch = random_of3_features(
            batch_size=batch_size, n_token=n_token,
            n_msa=consts.n_seq, n_templ=consts.n_templ,
        )
        batch = tensor_tree_map(lambda x: x.unsqueeze(1), batch)
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()

        xl_noisy = torch.randn((batch_size, 1, n_atom, 3))
        t = torch.ones(batch_size, 1)
        atom_mask = torch.ones((batch_size, 1, n_atom))
        si_input = torch.rand((batch_size, 1, n_token, c_s_input))
        si_trunk = torch.rand((batch_size, 1, n_token, c_s))
        zij_trunk = torch.rand((batch_size, 1, n_token, n_token, c_z))
        token_mask = batch["token_mask"]

        with torch.no_grad():
            conditioning_cache = dm.prepare_diffusion_conditioning_cache(
                batch=batch, si_input=si_input, si_trunk=si_trunk,
                zij_trunk=zij_trunk, use_conditioning=True,
            )
            atom_rep_cache = dm.prepare_atom_rep_cache(
                batch=batch, si_trunk=si_trunk,
                zij_conditioned=conditioning_cache["zij_conditioned"],
            )

            xl_uncached = dm(
                batch=batch, xl_noisy=xl_noisy, token_mask=token_mask,
                atom_mask=atom_mask, t=t, si_input=si_input, si_trunk=si_trunk,
                zij_trunk=zij_trunk, use_conditioning=True,
                conditioning_cache=conditioning_cache,
            )
            xl_cached = dm(
                batch=batch, xl_noisy=xl_noisy, token_mask=token_mask,
                atom_mask=atom_mask, t=t, si_input=si_input, si_trunk=si_trunk,
                zij_trunk=zij_trunk, use_conditioning=True,
                conditioning_cache=conditioning_cache,
                atom_rep_cache=atom_rep_cache,
            )
            torch.testing.assert_close(xl_cached, xl_uncached, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
