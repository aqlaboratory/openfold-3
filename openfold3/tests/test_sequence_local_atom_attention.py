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
import random
import unittest

import ml_collections as mlc
import torch

from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    NoisyPositionEmbedder,
    RefAtomFeatureEmbedder,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.tests.config import consts
from openfold3.tests.utils.compare_utils import assert_summation_order_close
from openfold3.tests.utils.data_utils import random_of3_features

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
            use_ada_layer_norm=True,
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
            use_ada_layer_norm=True,
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


class TestAtomAttentionEncoderAggregation(unittest.TestCase):
    """The encoder's eval-mode aggregation must be as repeatable as the train path.

    In eval mode (and only there) the encoder aggregates atom features to tokens
    with ``aggregate_atom_feat_to_tokens_segmented``. The training/autograd path
    cannot use it, because ``segment_reduce`` has no backward, so training keeps
    the scatter-based ``aggregate_atom_feat_to_tokens``. These two tests pin both
    halves of that arrangement: the eval path is bitwise repeatable, and it agrees
    numerically with the path training still uses.

    Note these tests deliberately do *not* take the ``device`` pytest fixture.
    That fixture turns on ``torch.use_deterministic_algorithms``, which makes
    ``scatter_add`` deterministic and would mask exactly the behaviour under test
    here.
    """

    @staticmethod
    def _build_encoder(batch_size, n_token, c_atom, c_token, seed, device):
        # ``random_of3_features`` draws from both the torch and the stdlib RNG
        # (``random_asym_ids`` uses ``random.randint``), so seed both or two calls
        # with the same seed still yield different batches.
        torch.manual_seed(seed)
        random.seed(seed)
        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )
        c_atom_pair = 8
        encoder = AtomAttentionEncoder(
            c_atom_ref=C_ATOM_REF,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=int(c_atom / 4),
            add_noisy_pos=False,
            no_heads=4,
            no_blocks=2,
            n_transition=2,
            n_query=32,
            n_key=128,
            use_ada_layer_norm=True,
        ).to(device)
        batch = tensor_tree_map(lambda t: t.to(device), batch)
        return encoder, batch

    @staticmethod
    def _forward(encoder, batch, training):
        """One encoder call; ``training`` selects scatter vs segmented aggregation."""
        previous = encoder.training
        encoder.train(training)
        try:
            with torch.no_grad():
                ai, _, _, _ = encoder(batch=batch)
        finally:
            encoder.train(previous)
        return ai

    def _run_with_consistent_forward(self, device, training):
        """Aggregate token features via one path, called 16 times.

        Returns the first result. Every later call must match it bitwise; the
        point of the test is whether it does, so any mismatch is the failure.
        """
        encoder, batch = self._build_encoder(
            batch_size=consts.batch_size,
            n_token=consts.n_res,
            c_atom=32,
            c_token=64,
            seed=0,
            device=device,
        )
        reference = self._forward(encoder, batch, training)
        for repeat in range(16):
            with self.subTest(device=device, repeat=repeat):
                self.assertTrue(
                    torch.equal(
                        self._forward(encoder, batch, training),
                        reference,
                    ),
                    "repeated encoder call is not bitwise identical",
                )
        return reference

    def test_eval_path_is_bitwise_repeatable(self):
        """Eval-mode (segmented) aggregation gives bitwise-identical repeats."""
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                self._run_with_consistent_forward(device, training=False)

    def test_eval_and_train_paths_agree(self):
        """Segmented (eval) and scatter (train) aggregation agree numerically.

        One encoder and one batch are reused for both calls, with only the
        ``training`` flag toggled, so the aggregation branch is the single
        difference between the two results.

        The tolerance is derived rather than chosen. The two paths sum each
        token's atoms in different orders, so the difference is bounded by the
        fp32 summation-order error over the atom count, not by any property of
        the output: see ``assert_summation_order_close``. ``atol`` alone cannot
        express that, because rounding-order error tracks the magnitude of the
        summed terms while ``assert_close`` scales with the output.
        """
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                encoder, batch = self._build_encoder(
                    batch_size=consts.batch_size,
                    n_token=consts.n_res,
                    c_atom=32,
                    c_token=64,
                    seed=0,
                    device=device,
                )

                segmented = self._forward(encoder, batch, training=False)
                scatter = self._forward(encoder, batch, training=True)

                assert_summation_order_close(
                    segmented,
                    scatter,
                    n_terms=int(batch["num_atoms_per_token"].max().item()),
                    msg=f"eval vs train aggregation on {device}",
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
            use_ada_layer_norm=True,
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


if __name__ == "__main__":
    unittest.main()
