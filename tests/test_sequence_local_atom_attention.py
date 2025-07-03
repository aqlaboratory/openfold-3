import math
import unittest

import torch

from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    NoisyPositionEmbedder,
    RefAtomFeatureEmbedder,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.config.base_config import c_atom_ref
from tests.config import consts
from tests.data_utils import random_af3_features


class TestRefAtomFeatureEmbedder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        c_atom = 64
        c_atom_pair = 16
        n_query = 32
        n_key = 128

        embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref.get(), c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = random_af3_features(
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
            c_atom_ref=c_atom_ref.get(), c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = random_af3_features(
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

        batch = random_af3_features(
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

        batch = random_af3_features(
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
            c_atom_ref=c_atom_ref.get(),
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
            use_ada_layer_norm=True,
            inf=inf,
        )

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        n_atom = batch["ref_pos"].shape[-2]

        num_blocks = math.ceil(n_atom / n_query)

        atom_mask = torch.ones((batch_size, n_atom))

        ai, ql, cl, plm = atom_attn_enc(batch=batch, atom_mask=atom_mask)

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
            c_atom_ref=c_atom_ref.get(),
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
            use_ada_layer_norm=True,
            inf=inf,
        )

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        atom_mask = torch.ones((batch_size, 1, n_atom))
        rl = torch.randn((batch_size, n_sample, n_atom, 3))
        si_trunk = torch.randn((batch_size, 1, n_token, c_s))
        zij_trunk = torch.randn((batch_size, 1, n_token, n_token, c_z))

        ai, ql, cl, plm = atom_attn_enc(
            batch=batch,
            atom_mask=atom_mask,
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
            use_ada_layer_norm=True,
            inf=inf,
        )

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        atom_mask = torch.ones((batch_size, n_atom))
        ai = torch.randn((batch_size, n_token, c_token))
        ql = torch.randn((batch_size, n_atom, c_atom))
        cl = torch.randn((batch_size, n_atom, c_atom))
        plm = torch.randn((batch_size, num_blocks, n_query, n_key, c_atom_pair))

        rl_update = atom_attn_dec(
            batch=batch, atom_mask=atom_mask, ai=ai, ql=ql, cl=cl, plm=plm
        )

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
            use_ada_layer_norm=True,
            inf=inf,
        )

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        n_atom = batch["ref_pos"].shape[-2]
        num_blocks = math.ceil(n_atom / n_query)

        atom_mask = torch.ones((batch_size, 1, n_atom))
        ai = torch.randn((batch_size, n_sample, n_token, c_token))
        ql = torch.randn((batch_size, n_sample, n_atom, c_atom))
        cl = torch.randn((batch_size, 1, n_atom, c_atom))
        plm = torch.randn((batch_size, 1, num_blocks, n_query, n_key, c_atom_pair))

        rl_update = atom_attn_dec(
            batch=batch, atom_mask=atom_mask, ai=ai, ql=ql, cl=cl, plm=plm
        )

        self.assertTrue(rl_update.shape == (batch_size, n_sample, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
