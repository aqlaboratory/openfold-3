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

import math
import unittest

import ml_collections as mlc
import torch

from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    NoisyPositionEmbedder,
    RefAtomFeatureEmbedder,
)
from openfold3.core.utils.atom_attention_block_utils import (
    convert_single_rep_to_blocks,
    get_atom_pair_block_mask,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import random_of3_features, randomize_parameters

C_ATOM_REF = mlc.ConfigDict(
    {
        "element": 119,
        "name_chars": 256,
    }
)


class TestRefAtomFeatureEmbedder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        c_atom = 64
        c_atom_pair = 16
        n_query = 32
        n_key = 128

        embedder = RefAtomFeatureEmbedder(
            c_atom_ref=C_ATOM_REF, c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=consts.n_res,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
            is_eval=False,
        )

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        cl, plm = embedder(batch, n_query=n_query, n_key=n_key)

        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, num_blocks, n_query, n_key, c_atom_pair)
        )

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        c_atom = 64
        c_atom_pair = 16
        n_query = 32
        n_key = 128

        embedder = RefAtomFeatureEmbedder(
            c_atom_ref=C_ATOM_REF, c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=consts.n_res,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
            is_eval=False,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        cl, plm = embedder(batch, n_query=n_query, n_key=n_key)

        self.assertTrue(cl.shape == (batch_size, 1, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, 1, num_blocks, n_query, n_key, c_atom_pair)
        )


class TestNoisyPositionEmbedder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 64
        c_atom_pair = 16
        n_query = 32
        n_key = 128

        embedder = NoisyPositionEmbedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
            is_eval=False,
        )

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        cl = torch.randn((batch_size, n_atom, c_atom))
        plm = torch.randn((batch_size, num_blocks, n_query, n_key, c_atom_pair))

        si_trunk = torch.randn((batch_size, n_token, c_s))
        zij_trunk = torch.randn((batch_size, n_token, n_token, c_z))
        rl = torch.randn((batch_size, n_atom, 3))

        cl, plm, ql = embedder(
            batch=batch,
            cl=cl,
            plm=plm,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            rl=rl,
            n_query=n_query,
            n_key=n_key,
        )

        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, num_blocks, n_query, n_key, c_atom_pair)
        )
        self.assertTrue(ql.shape == (batch_size, n_atom, c_atom))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 64
        c_atom_pair = 16
        n_sample = 3
        n_query = 32
        n_key = 128

        embedder = NoisyPositionEmbedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
            is_eval=False,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        cl = torch.randn((batch_size, 1, n_atom, c_atom))
        plm = torch.randn((batch_size, 1, num_blocks, n_query, n_key, c_atom_pair))

        si_trunk = torch.randn((batch_size, 1, n_token, c_s))
        zij_trunk = torch.randn((batch_size, 1, n_token, n_token, c_z))
        rl = torch.randn((batch_size, n_sample, n_atom, 3))

        cl, plm, ql = embedder(
            batch=batch,
            cl=cl,
            plm=plm,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            rl=rl,
            n_query=n_query,
            n_key=n_key,
        )

        self.assertTrue(cl.shape == (batch_size, 1, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, 1, num_blocks, n_query, n_key, c_atom_pair)
        )
        self.assertTrue(ql.shape == (batch_size, n_sample, n_atom, c_atom))


class TestAtomAttentionEncoder(unittest.TestCase):
    def test_without_noisy_positions(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_atom = 128
        c_atom_pair = 16
        c_token = 384
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        c_hidden = int(c_atom / no_heads)
        n_query = 32
        n_key = 128
        inf = 1e10

        atom_attn_enc = AtomAttentionEncoder(
            c_atom_ref=C_ATOM_REF,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=c_hidden,
            add_noisy_pos=False,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        n_atom = batch["ref_pos"].shape[-2]

        num_blocks = math.ceil(n_atom / n_query)

        ai, ql, cl, plm = atom_attn_enc(batch=batch)

        self.assertTrue(ai.shape == (batch_size, n_token, c_token))
        self.assertTrue(ql.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, num_blocks, n_query, n_key, c_atom_pair)
        )

    def test_with_noisy_positions(self):
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
        c_hidden = int(c_atom / no_heads)
        n_query = 32
        n_key = 128
        inf = 1e10
        n_sample = 3

        atom_attn_enc = AtomAttentionEncoder(
            c_s=c_s,
            c_z=c_z,
            c_atom_ref=C_ATOM_REF,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=c_hidden,
            add_noisy_pos=True,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        rl = torch.randn((batch_size, n_sample, n_atom, 3))
        si_trunk = torch.randn((batch_size, 1, n_token, c_s))
        zij_trunk = torch.randn((batch_size, 1, n_token, n_token, c_z))

        ai, ql, cl, plm = atom_attn_enc(
            batch=batch,
            rl=rl,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )

        self.assertTrue(ai.shape == (batch_size, n_sample, n_token, c_token))
        self.assertTrue(ql.shape == (batch_size, n_sample, n_atom, c_atom))
        self.assertTrue(cl.shape == (batch_size, 1, n_atom, c_atom))
        self.assertTrue(
            plm.shape == (batch_size, 1, num_blocks, n_query, n_key, c_atom_pair)
        )


class TestAtomAttentionDecoder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_atom = 128
        c_atom_pair = 16
        c_token = 384
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        c_hidden = int(c_atom / no_heads)
        n_query = 32
        n_key = 128
        inf = 1e10

        atom_attn_dec = AtomAttentionDecoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        ai = torch.randn((batch_size, n_token, c_token))
        ql = torch.randn((batch_size, n_atom, c_atom))
        cl = torch.randn((batch_size, n_atom, c_atom))
        plm = torch.randn((batch_size, num_blocks, n_query, n_key, c_atom_pair))

        rl_update = atom_attn_dec(batch=batch, ai=ai, ql=ql, cl=cl, plm=plm)

        self.assertTrue(rl_update.shape == (batch_size, n_atom, 3))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_atom = 128
        c_atom_pair = 16
        c_token = 384
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        c_hidden = int(c_atom / no_heads)
        n_query = 32
        n_key = 128
        inf = 1e10
        n_sample = 3

        atom_attn_dec = AtomAttentionDecoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        ai = torch.randn((batch_size, n_sample, n_token, c_token))
        ql = torch.randn((batch_size, n_sample, n_atom, c_atom))
        cl = torch.randn((batch_size, 1, n_atom, c_atom))
        plm = torch.randn((batch_size, 1, num_blocks, n_query, n_key, c_atom_pair))

        rl_update = atom_attn_dec(batch=batch, ai=ai, ql=ql, cl=cl, plm=plm)

        self.assertTrue(rl_update.shape == (batch_size, n_sample, n_atom, 3))


class TestAtomAttentionStepInvariants(unittest.TestCase):
    """
    `AtomAttentionEncoder.get_atom_conditioning` factors out the part of the encoder
    that does not depend on the noisy positions, so it (and the attention mask bias)
    can be precomputed once per diffusion rollout. Using the precomputed values via
    `atom_cond` / `mask_bias` must give the same result as the default path.
    """

    def test_encoder_and_decoder_with_precomputed_inputs(self):
        # The default path computes the encoder outputs from the noisy positions, so the
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 32
        c_atom_pair = 8
        c_token = 48
        no_heads = 4
        n_query = 32
        n_key = 128
        n_sample = 3

        common = {
            "c_atom": c_atom,
            "c_atom_pair": c_atom_pair,
            "c_token": c_token,
            "c_hidden": c_atom // no_heads,
            "no_heads": no_heads,
            "no_blocks": 2,
            "n_transition": 2,
            "n_query": n_query,
            "n_key": n_key,
            "inf": 1e9,
        }
        atom_attn_enc = randomize_parameters(
            AtomAttentionEncoder(
                c_s=c_s,
                c_z=c_z,
                c_atom_ref=C_ATOM_REF,
                add_noisy_pos=True,
                **common,
            ).eval()
        )
        atom_attn_dec = randomize_parameters(AtomAttentionDecoder(**common).eval())

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        n_atom = batch["ref_pos"].shape[-2]
        # Padded atoms, so that the attention mask bias is not trivially zero
        batch["atom_mask"][..., -7:] = 0

        rl = torch.randn((batch_size, n_sample, n_atom, 3))
        si_trunk = torch.randn((batch_size, 1, n_token, c_s))
        zij_trunk = torch.randn((batch_size, 1, n_token, n_token, c_z))

        with torch.no_grad():
            # The default path computes the encoder outputs from the noisy positions
            ai_exp, ql_exp, cl_exp, plm_exp = atom_attn_enc(
                batch=batch, rl=rl, si_trunk=si_trunk, zij_trunk=zij_trunk
            )
            rl_update_exp = atom_attn_dec(
                batch=batch, ai=ai_exp, ql=ql_exp, cl=cl_exp, plm=plm_exp
            )

            atom_cond = atom_attn_enc.get_atom_conditioning(
                batch=batch, si_trunk=si_trunk, zij_trunk=zij_trunk
            )
            enc_mask_bias = atom_attn_enc.atom_transformer.get_mask_bias(
                batch["atom_mask"]
            )
            dec_mask_bias = atom_attn_dec.atom_transformer.get_mask_bias(
                batch["atom_mask"]
            )

            # The trunk inputs are not needed when atom_cond is given
            ai, ql, cl, plm = atom_attn_enc(
                batch=batch, rl=rl, atom_cond=atom_cond, mask_bias=enc_mask_bias
            )
            rl_update = atom_attn_dec(
                batch=batch, ai=ai, ql=ql, cl=cl, plm=plm, mask_bias=dec_mask_bias
            )

        self.assertIs(cl, atom_cond[0])
        self.assertIs(plm, atom_cond[1])
        for actual, expected in (
            (ai, ai_exp),
            (ql, ql_exp),
            (cl, cl_exp),
            (plm, plm_exp),
            (rl_update, rl_update_exp),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_atom_conditioning_does_not_depend_on_noisy_positions(self):
        # The outputs of `get_atom_conditioning` should not depend on the noisy positions, so calling it with different noisy positions should give the same result.
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z

        atom_attn_enc = randomize_parameters(
            AtomAttentionEncoder(
                c_s=c_s,
                c_z=c_z,
                c_atom_ref=C_ATOM_REF,
                c_atom=32,
                c_atom_pair=8,
                c_token=48,
                c_hidden=8,
                add_noisy_pos=True,
                no_heads=4,
                no_blocks=1,
                n_transition=2,
                n_query=32,
                n_key=128,
            ).eval()
        )

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        n_atom = batch["ref_pos"].shape[-2]

        si_trunk = torch.randn((batch_size, n_token, c_s))
        zij_trunk = torch.randn((batch_size, n_token, n_token, c_z))

        with torch.no_grad():
            cl, plm = atom_attn_enc.get_atom_conditioning(
                batch=batch, si_trunk=si_trunk, zij_trunk=zij_trunk
            )
            # cl and plm returned by the encoder are the conditioning, whatever rl is
            for _ in range(2):
                _, _, cl_out, plm_out = atom_attn_enc(
                    batch=batch,
                    rl=torch.randn((batch_size, n_atom, 3)),
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk,
                )
                torch.testing.assert_close(cl_out, cl, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(plm_out, plm, rtol=1e-6, atol=1e-6)


class TestBlockMaskHelpers(unittest.TestCase):
    """
    `get_atom_pair_block_mask` factors the 2D q/k block mask out of
    `convert_single_rep_to_blocks`, so the mask (and the attention bias built from
    it) can be precomputed once per rollout instead of alongside every gather of the
    (per-step) single representation.
    """

    def setUp(self):
        # Setup parameters for the tests, part of unittest library, called before each test method
        self.n_query = 32
        self.n_key = 128
        self.n_atom = 150

        self.atom_mask = torch.ones((consts.batch_size, 1, self.n_atom))
        self.atom_mask[..., -20:] = 0

    def test_get_atom_pair_block_mask_matches_convert_single_rep_to_blocks(self):
        # The 2D q/k block mask computed by `get_atom_pair_block_mask` should match the one computed by `convert_single_rep_to_blocks`, when the latter is called with `compute_pair_mask=True`.
        n_sample = 3
        ql = torch.randn((consts.batch_size, n_sample, self.n_atom, 4))

        _, _, expected = convert_single_rep_to_blocks(
            ql=ql, n_query=self.n_query, n_key=self.n_key, atom_mask=self.atom_mask
        )
        actual = get_atom_pair_block_mask(
            atom_mask=self.atom_mask, n_query=self.n_query, n_key=self.n_key
        )

        # The standalone mask is computed once for the sample dim of the mask,
        # and broadcasts against the per-sample mask
        self.assertEqual(actual.shape[1], 1)
        torch.testing.assert_close(actual.expand_as(expected), expected)

    def test_convert_single_rep_to_blocks_without_pair_mask(self):
        # The `compute_pair_mask` argument of `convert_single_rep_to_blocks` should control whether the 2D q/k block mask is returned or not.
        ql = torch.randn((consts.batch_size, 1, self.n_atom, 4))

        q_exp, k_exp, _ = convert_single_rep_to_blocks(
            ql=ql, n_query=self.n_query, n_key=self.n_key, atom_mask=self.atom_mask
        )
        q, k, pair_mask = convert_single_rep_to_blocks(
            ql=ql,
            n_query=self.n_query,
            n_key=self.n_key,
            atom_mask=self.atom_mask,
            compute_pair_mask=False,
        )

        self.assertIsNone(pair_mask)
        torch.testing.assert_close(q, q_exp)
        torch.testing.assert_close(k, k_exp)


if __name__ == "__main__":
    unittest.main()
