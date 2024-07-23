import unittest
from contextlib import nullcontext

import torch

import tests.compare_utils as compare_utils
from openfold3.core.model.layers import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    AtomTransformer,
    NoisyPositionEmbedder,
    RefAtomFeatureEmbedder,
)
from openfold3.core.model.primitives.initialization import lecun_normal_init_
from openfold3.model_implementations.af3_all_atom.config import c_atom_ref
from tests.config import consts


class TestRefAtomFeatureEmbedder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_atom = 4 * consts.n_res
        c_atom = 64
        c_atom_pair = 16

        embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref.get(), c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = {
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
        }

        cl, plm = embedder(batch)

        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(plm.shape == (batch_size, n_atom, n_atom, c_atom_pair))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_atom = 4 * consts.n_res
        c_atom = 64
        c_atom_pair = 16

        embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref.get(), c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        batch = {
            "ref_pos": torch.randn((batch_size, 1, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, 1, n_atom)),
            "ref_element": torch.ones((batch_size, 1, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, 1, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, 1, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, 1, n_atom)),
        }

        cl, plm = embedder(batch)

        self.assertTrue(cl.shape == (batch_size, 1, n_atom, c_atom))
        self.assertTrue(plm.shape == (batch_size, 1, n_atom, n_atom, c_atom_pair))


class TestNoisyPositionEmbedder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 64
        c_atom_pair = 16

        embedder = NoisyPositionEmbedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        cl = torch.ones((batch_size, n_atom, c_atom))
        plm = torch.ones((batch_size, n_atom, n_atom, c_atom_pair))
        ql = torch.ones((batch_size, n_atom, c_atom))

        si_trunk = torch.ones((batch_size, n_token, c_s))
        zij_trunk = torch.ones((batch_size, n_token, n_token, c_z))
        rl = torch.randn((batch_size, n_atom, 3))

        batch = {"atom_to_token_index": torch.ones((batch_size, n_atom))}

        cl, plm, ql = embedder(
            batch=batch,
            cl=cl,
            plm=plm,
            ql=ql,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            rl=rl,
        )

        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(plm.shape == (batch_size, n_atom, n_atom, c_atom_pair))
        self.assertTrue(ql.shape == (batch_size, n_atom, c_atom))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_atom = 64
        c_atom_pair = 16
        n_sample = 3

        embedder = NoisyPositionEmbedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        cl = torch.ones((batch_size, 1, n_atom, c_atom))
        plm = torch.ones((batch_size, 1, n_atom, n_atom, c_atom_pair))
        ql = torch.ones((batch_size, 1, n_atom, c_atom))

        si_trunk = torch.ones((batch_size, 1, n_token, c_s))
        zij_trunk = torch.ones((batch_size, 1, n_token, n_token, c_z))
        rl = torch.randn((batch_size, n_sample, n_atom, 3))

        batch = {"atom_to_token_index": torch.ones((batch_size, 1, n_atom))}

        cl, plm, ql = embedder(
            batch=batch,
            cl=cl,
            plm=plm,
            ql=ql,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            rl=rl,
        )

        self.assertTrue(cl.shape == (batch_size, 1, n_atom, c_atom))
        self.assertTrue(plm.shape == (batch_size, 1, n_atom, n_atom, c_atom_pair))
        self.assertTrue(ql.shape == (batch_size, n_sample, n_atom, c_atom))


class TestAtomTransformer(unittest.TestCase):
    def run_shape_test(
        self, ql, cl, plm, atom_mask, out_shape, dtype, use_block_sparse_attn=False
    ):
        c_atom = 128
        c_atom_pair = 16
        c_hidden = 32
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        n_query = 32
        n_key = 128
        block_size = 16
        inf = 1e10
        device = "cuda" if torch.cuda.is_available() else "cpu"

        atom_transformer = (
            AtomTransformer(
                c_q=c_atom,
                c_p=c_atom_pair,
                c_hidden=c_hidden,
                no_heads=no_heads,
                no_blocks=no_blocks,
                n_transition=n_transition,
                n_query=n_query,
                n_key=n_key,
                use_ada_layer_norm=True,
                use_block_sparse_attn=use_block_sparse_attn,
                block_size=block_size,
                inf=inf,
            )
            .eval()
            .to(device)
        )

        ql = ql.to(device, dtype=dtype)
        cl = cl.to(device, dtype=dtype)
        plm = plm.to(device, dtype=dtype)
        atom_mask = atom_mask.to(device, dtype=dtype)

        cuda_context = (
            torch.cuda.amp.autocast(dtype=dtype)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with cuda_context:
            ql = atom_transformer(ql=ql, cl=cl, plm=plm, atom_mask=atom_mask).cpu()

        self.assertTrue(ql.shape == out_shape)

    def without_n_sample_channel(self, dtype, use_block_sparse_attn):
        batch_size = consts.batch_size
        n_token = 192  # Has to be divisible by block size
        n_atom = 4 * n_token
        c_atom = 128
        c_atom_pair = 16

        ql = torch.ones((batch_size, n_atom, c_atom))
        cl = torch.ones((batch_size, n_atom, c_atom))
        plm = torch.ones((batch_size, n_atom, n_atom, c_atom_pair))
        atom_mask = torch.ones((batch_size, n_atom))

        out_shape = (batch_size, n_atom, c_atom)

        self.run_shape_test(
            ql, cl, plm, atom_mask, out_shape, dtype, use_block_sparse_attn
        )

    def with_n_sample_channel(self, dtype, use_block_sparse_attn):
        batch_size = consts.batch_size
        n_token = 192  # Has to be divisible by block size
        n_atom = 4 * n_token
        c_atom = 128
        c_atom_pair = 16
        n_sample = 3

        ql = torch.ones((batch_size, n_sample, n_atom, c_atom))
        cl = torch.ones((batch_size, 1, n_atom, c_atom))
        plm = torch.ones((batch_size, 1, n_atom, n_atom, c_atom_pair))
        atom_mask = torch.ones((batch_size, 1, n_atom))

        out_shape = (batch_size, n_sample, n_atom, c_atom)

        self.run_shape_test(
            ql, cl, plm, atom_mask, out_shape, dtype, use_block_sparse_attn
        )

    def test_without_block_sparse_attn(self):
        self.without_n_sample_channel(dtype=torch.float32, use_block_sparse_attn=False)
        self.with_n_sample_channel(dtype=torch.float32, use_block_sparse_attn=False)

        if torch.cuda.is_available():
            self.without_n_sample_channel(
                dtype=torch.bfloat16, use_block_sparse_attn=False
            )
            self.with_n_sample_channel(
                dtype=torch.bfloat16, use_block_sparse_attn=False
            )

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_with_block_sparse_attn(self):
        self.without_n_sample_channel(dtype=torch.float32, use_block_sparse_attn=True)
        self.with_n_sample_channel(dtype=torch.float32, use_block_sparse_attn=True)

        if torch.cuda.is_available():
            self.without_n_sample_channel(
                dtype=torch.bfloat16, use_block_sparse_attn=True
            )
            self.with_n_sample_channel(dtype=torch.bfloat16, use_block_sparse_attn=True)

    def compare_block_sparse(self, dtype):
        batch_size = consts.batch_size
        n_token = 192  # Has to be divisible by block size
        n_atom = 4 * n_token
        n_sample = 3
        c_atom = 128
        c_atom_pair = 16
        c_hidden = 32
        no_heads = 4
        no_blocks = 3
        n_transition = 2
        n_query = 32
        n_key = 128
        block_size = 16
        inf = 1e10
        eps = consts.eps
        device = "cuda" if torch.cuda.is_available() else "cpu"

        atom_transformer = (
            AtomTransformer(
                c_q=c_atom,
                c_p=c_atom_pair,
                c_hidden=c_hidden,
                no_heads=no_heads,
                no_blocks=no_blocks,
                n_transition=n_transition,
                n_query=n_query,
                n_key=n_key,
                use_ada_layer_norm=True,
                use_block_sparse_attn=False,
                block_size=block_size,
                inf=inf,
            )
            .eval()
            .to(device)
        )

        ql = torch.ones((batch_size, n_sample, n_atom, c_atom)).to(device, dtype=dtype)
        cl = torch.ones((batch_size, 1, n_atom, c_atom)).to(device, dtype=dtype)
        plm = torch.ones((batch_size, 1, n_atom, n_atom, c_atom_pair)).to(
            device, dtype=dtype
        )
        atom_mask = torch.ones((batch_size, 1, n_atom)).to(device, dtype=dtype)

        with torch.no_grad():
            for i in range(no_blocks):
                apb = atom_transformer.diffusion_transformer.blocks[
                    i
                ].attention_pair_bias
                lecun_normal_init_(apb.mha.linear_g.weight)
                lecun_normal_init_(apb.mha.linear_o.weight)
                lecun_normal_init_(apb.linear_ada_out.weight)

            cuda_context = (
                torch.cuda.amp.autocast(dtype=dtype)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with cuda_context:
                ql_out = atom_transformer(
                    ql=ql, cl=cl, plm=plm, atom_mask=atom_mask
                ).cpu()
                atom_transformer.use_block_sparse_attn = True
                ql_out_block_sparse = atom_transformer(
                    ql=ql, cl=cl, plm=plm, atom_mask=atom_mask
                ).cpu()
                err = torch.mean(torch.abs(ql_out - ql_out_block_sparse))
                self.assertTrue(err < eps, f"Error: {err}")

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_compare_block_sparse_fp32(self):
        self.compare_block_sparse(dtype=torch.float32)

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_compare_block_sparse_bf16(self):
        self.compare_block_sparse(dtype=torch.bfloat16)


class TestAtomAttentionEncoder(unittest.TestCase):
    def test_without_noisy_positions(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
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
            use_block_sparse_attn=False,
            block_size=None,
            inf=inf,
        )

        batch = {
            "token_mask": torch.ones((batch_size, n_token)),
            "atom_to_token_index": torch.ones((batch_size, n_atom)),
            "ref_pos": torch.randn((batch_size, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, n_atom)),
            "ref_element": torch.ones((batch_size, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, n_atom)),
        }

        atom_mask = torch.ones((batch_size, n_atom))

        ai, ql, cl, plm = atom_attn_enc(batch=batch, atom_mask=atom_mask)

        self.assertTrue(ai.shape == (batch_size, n_token, c_token))
        self.assertTrue(ql.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        self.assertTrue(plm.shape == (batch_size, n_atom, n_atom, c_atom_pair))

    def test_with_noisy_positions(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
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
            use_block_sparse_attn=False,
            block_size=None,
            inf=inf,
        )

        batch = {
            "token_mask": torch.ones((batch_size, 1, n_token)),
            "atom_to_token_index": torch.ones((batch_size, 1, n_atom)),
            "ref_pos": torch.randn((batch_size, 1, n_atom, 3)),
            "ref_mask": torch.ones((batch_size, 1, n_atom)),
            "ref_element": torch.ones((batch_size, 1, n_atom, 128)),
            "ref_charge": torch.ones((batch_size, 1, n_atom)),
            "ref_atom_name_chars": torch.ones((batch_size, 1, n_atom, 4, 64)),
            "ref_space_uid": torch.zeros((batch_size, 1, n_atom)),
        }

        atom_mask = torch.ones((batch_size, 1, n_atom))
        rl = torch.randn((batch_size, n_sample, n_atom, 3))
        si_trunk = torch.ones((batch_size, 1, n_token, c_s))
        zij_trunk = torch.ones((batch_size, 1, n_token, n_token, c_z))

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
        self.assertTrue(plm.shape == (batch_size, 1, n_atom, n_atom, c_atom_pair))


class TestAtomAttentionDecoder(unittest.TestCase):
    def test_without_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
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
            use_block_sparse_attn=False,
            block_size=None,
            inf=inf,
        )

        batch = {
            "atom_to_token_index": torch.ones((batch_size, n_atom)),
        }

        atom_mask = torch.ones((batch_size, n_atom))
        ai = torch.ones((batch_size, n_token, c_token))
        ql = torch.ones((batch_size, n_atom, c_atom))
        cl = torch.ones((batch_size, n_atom, c_atom))
        plm = torch.ones((batch_size, n_atom, n_atom, c_atom_pair))

        rl_update = atom_attn_dec(
            batch=batch, atom_mask=atom_mask, ai=ai, ql=ql, cl=cl, plm=plm
        )

        self.assertTrue(rl_update.shape == (batch_size, n_atom, 3))

    def test_with_n_sample_channel(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
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
            use_block_sparse_attn=False,
            block_size=None,
            inf=inf,
        )

        batch = {
            "atom_to_token_index": torch.ones((batch_size, 1, n_atom)),
        }

        atom_mask = torch.ones((batch_size, 1, n_atom))
        ai = torch.ones((batch_size, n_sample, n_token, c_token))
        ql = torch.ones((batch_size, n_sample, n_atom, c_atom))
        cl = torch.ones((batch_size, 1, n_atom, c_atom))
        plm = torch.ones((batch_size, 1, n_atom, n_atom, c_atom_pair))

        rl_update = atom_attn_dec(
            batch=batch, atom_mask=atom_mask, ai=ai, ql=ql, cl=cl, plm=plm
        )

        self.assertTrue(rl_update.shape == (batch_size, n_sample, n_atom, 3))


if __name__ == "__main__":
    unittest.main()
