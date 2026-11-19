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

import pytest
import torch

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.model.primitives.attention import (
    Attention,
    _attention,
    _sdpa_attention,
    _sdpa_is_eligible,
)
from openfold3.core.model.primitives.initialization import lecun_normal_init_
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import random_attention_inputs


class TestLMA(unittest.TestCase):
    @pytest.mark.slow
    @compare_utils.skip_unless_gpu_available()
    def test_lma_vs_attention(self):
        c_hidden = 32
        no_heads = 4

        q, kv, _, biases = random_attention_inputs(
            batch_size=consts.batch_size,
            n_seq=consts.n_seq,
            n=2**12,
            no_heads=no_heads,
            c_hidden=c_hidden,
        )

        a = Attention(
            c_hidden,
            c_hidden,
            c_hidden,
            c_hidden,
            no_heads,
        ).cuda()

        with torch.no_grad():
            lecun_normal_init_(a.linear_g.weight)
            lecun_normal_init_(a.linear_o.weight)

            l = a(q, kv, biases=biases, use_lma=True).cpu()
            real = a(q, kv, biases=biases).cpu()

        err = torch.max(torch.abs(l - real))
        self.assertTrue(err < consts.eps, f"Error: {err}")


class TestSDPA(unittest.TestCase):
    """``_sdpa_attention`` must match ``_attention`` for every bias layout.

    The bias batch dims are the interesting axis: SDPA accepts a leading mask
    dim of 1 or exactly the flattened batch size, and nothing in between. A
    partially broadcast bias -- ``(B, 1, H, N, N)`` against ``(B, S, H, N, N)``
    queries, which is what ``random_attention_inputs`` builds -- folds to ``B``
    and used to raise rather than fall back.
    """

    B, S, H, N, C = consts.batch_size, consts.n_seq, 4, 16, 32

    def _run(self, bias_batch, mask_batch=(1, 1), use_high_precision=True):
        torch.manual_seed(0)
        q = torch.randn((self.B, self.S, self.H, self.N, self.C), device="cuda")
        q = q / self.C**0.5
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        z = 0.1 * torch.randn((*bias_batch, self.H, self.N, self.N), device="cuda")
        mask = torch.randint(0, 2, (*mask_batch, 1, 1, self.N), device="cuda").float()
        biases = [1e9 * (mask - 1), z]

        self.assertTrue(_sdpa_is_eligible(q, k, v, biases))

        with torch.no_grad():
            ref = _attention(q, k, v, biases, use_high_precision=use_high_precision)
            out = _sdpa_attention(q, k, v, biases)

        err = (out - ref).abs().max().item()
        self.assertTrue(err < consts.eps, f"bias_batch={bias_batch}: {err}")

    def test_shared_pair_bias(self):
        self._run((1, 1))

    def test_fully_batched_pair_bias(self):
        self._run((self.B, self.S))

    def test_partially_broadcast_pair_bias(self):
        # This is the ``random_attention_inputs`` layout, and the case that
        # raised before the fold learned to expand shared dims.
        self._run((self.B, 1))

    def test_partially_broadcast_other_axis(self):
        self._run((1, self.S))

    def test_bias_with_fewer_batch_dims(self):
        # A bias may carry *fewer* batch dims than the query and still
        # broadcast, since broadcasting right-aligns them. Gating the fold on
        # ``mask.dim() > 4`` skipped this shape, so it reached SDPA with a
        # batch dim of ``S`` that cannot expand to ``B * S`` and raised.
        torch.manual_seed(0)
        q = torch.randn((self.B, self.S, self.H, self.N, self.C), device="cuda")
        q = q / self.C**0.5
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # [S, H, N, N]: no batch dim, one shared-by-batch axis.
        biases = [0.1 * torch.randn((self.S, self.H, self.N, self.N), device="cuda")]

        self.assertTrue(_sdpa_is_eligible(q, k, v, biases))

        with torch.no_grad():
            ref = _attention(q, k, v, biases, use_high_precision=True)
            out = _sdpa_attention(q, k, v, biases)

        err = (out - ref).abs().max().item()
        self.assertTrue(err < consts.eps, f"fewer batch dims: {err}")

    def test_four_dimensional_query_is_not_eligible(self):
        # Already flat: the fold is skipped, so the bias must already match.
        q = torch.randn((self.H, self.N, self.C), device="cuda")
        biases = [torch.zeros((1, 1, 1, self.N), device="cuda")]
        self.assertFalse(_sdpa_is_eligible(q, q, q, biases))


if __name__ == "__main__":
    unittest.main()
