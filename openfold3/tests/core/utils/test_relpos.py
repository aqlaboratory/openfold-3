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

import pytest
import torch

from openfold3.core.utils.relpos import cyclic_offset, relpos_complex


def _make_batch(n_token, asym_ids, cyclic_mask, batch_size=1):
    """Build a minimal feature dict for relpos_complex."""
    residue_index = torch.arange(n_token).unsqueeze(0).repeat(batch_size, 1)
    token_index = torch.arange(n_token).unsqueeze(0).repeat(batch_size, 1)
    asym_id = (
        torch.tensor(asym_ids, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1)
    )
    entity_id = asym_id.clone()
    sym_id = torch.ones(batch_size, n_token, dtype=torch.int32)
    cm = torch.tensor(cyclic_mask, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1)
    return {
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "cyclic_mask": cm,
    }


class TestCyclicOffset:
    def test_diagonal_is_zero(self):
        # A residue's distance to itself is always 0.
        idx = torch.arange(8)
        off = cyclic_offset(idx)
        assert (off.diagonal() == 0).all()

    def test_antisymmetry(self):
        # Cyclic offset is antisymmetric: off[i,j] == -off[j,i].
        # Equivalently, the magnitudes are symmetric.
        idx = torch.arange(6)
        off = cyclic_offset(idx)
        assert torch.equal(off.abs(), off.T.abs())

    def test_max_distance_at_midpoint(self):
        # For an even-length chain the maximum offset magnitude is peptide_length // 2.
        n = 10
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        assert int(off.abs().max()) == n // 2

    def test_odd_length(self):
        # For an odd-length chain all entries are <= (n-1)//2 away from 0.
        n = 7
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        assert int(off.abs().max()) <= (n - 1) // 2 + 1

    def test_output_shape(self):
        n = 5
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        assert off.shape == (n, n)

    def test_values_wrap_correctly(self):
        # For n=6 the cyclic row starting at 0 should be:
        #   [0, -1, -2, -3, 2, 1]
        # i.e. going forward costs +dist, wrapping back costs negative dist past midpoint.
        n = 6
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        row0 = off[0].tolist()
        # Distance from 0 to 3 (midpoint) is -3; to 4 wraps back: +2; to 5: +1.
        assert row0[0] == 0
        assert row0[1] == -1
        assert row0[2] == -2
        assert row0[3] == -3
        assert row0[4] == 2
        assert row0[5] == 1


class TestRelposComplex:
    MAX_IDX = 32
    MAX_CHAIN = 2

    def _relpos(self, batch):
        return relpos_complex(batch, self.MAX_IDX, self.MAX_CHAIN)

    @pytest.mark.parametrize("is_cyclic", [True, False])
    def test_relpos_shape(self, is_cyclic):
        n = 10
        batch = _make_batch(n, [1] * n, [is_cyclic] * n)
        out = self._relpos(batch)
        expected_last = (2 * self.MAX_IDX + 2) * 2 + 1 + (2 * self.MAX_CHAIN + 2)
        assert out.shape == (1, n, n, expected_last)

    def test_cyclic_changes_encoding_vs_linear(self):
        # A cyclic chain should produce different rel-pos encodings than a linear one.
        n = 10
        linear_batch = _make_batch(n, [1] * n, [False] * n)
        cyclic_batch = _make_batch(n, [1] * n, [True] * n)
        linear_out = self._relpos(linear_batch)
        cyclic_out = self._relpos(cyclic_batch)
        assert not torch.equal(linear_out, cyclic_out)

    def test_cyclic_self_pairs_get_center_bin(self):
        # Self-pairs always have offset=0, which clamps to MAX_IDX → one-hot at bin MAX_IDX.
        n = 8
        batch = _make_batch(n, [1] * n, [True] * n)
        out = self._relpos(batch)
        rel_pos_slice = out[0, :, :, : 2 * self.MAX_IDX + 2]
        diag = torch.stack([rel_pos_slice[i, i] for i in range(n)])
        assert (diag[:, self.MAX_IDX] == 1.0).all()
        assert (diag[:, self.MAX_IDX] == diag.sum(dim=-1)).all()

    def test_non_cyclic_chain_unchanged_in_multimer(self):
        # In a multimer where only chain 2 is cyclic, chain 1's encoding should
        # be identical to an all-linear batch.
        n_lin = 6
        n_cyc = 6
        n = n_lin + n_cyc
        asym_ids = [1] * n_lin + [2] * n_cyc
        cyclic_mask_mixed = [False] * n_lin + [True] * n_cyc
        cyclic_mask_none = [False] * n

        mixed_batch = _make_batch(n, asym_ids, cyclic_mask_mixed)
        linear_batch = _make_batch(n, asym_ids, cyclic_mask_none)

        mixed_out = self._relpos(mixed_batch)
        linear_out = self._relpos(linear_batch)

        # Chain 1 rows/cols (indices 0..n_lin-1) should be unchanged.
        ch1_mixed = mixed_out[0, :n_lin, :n_lin, :]
        ch1_linear = linear_out[0, :n_lin, :n_lin, :]
        assert torch.equal(ch1_mixed, ch1_linear)

    def test_cross_chain_pairs_unaffected_by_cyclic(self):
        # Cross-chain token pairs should use the "different chain" sentinel regardless
        # of whether either chain is cyclic (same_chain=False → clipped to sentinel).
        n_a = 5
        n_b = 5
        n = n_a + n_b
        asym_ids = [1] * n_a + [2] * n_b
        cyclic_mask = [True] * n_a + [False] * n_b

        batch = _make_batch(n, asym_ids, cyclic_mask)
        out = self._relpos(batch)

        # The cross-chain rel_pos block uses the sentinel bin (2*MAX_IDX+1).
        # After one-hot encoding each row sums to 1; check the sentinel column.
        rel_pos_slice = out[0, :n_a, n_a:, : 2 * self.MAX_IDX + 2]
        sentinel_col = 2 * self.MAX_IDX + 1
        assert (rel_pos_slice[..., sentinel_col] == 1).all()

    def test_no_cyclic_mask_does_not_raise(self):
        # cyclic_mask all-False should run without error and match linear baseline.
        n = 8
        batch_false = _make_batch(n, [1] * n, [False] * n)
        batch_zeros = _make_batch(n, [1] * n, [False] * n)
        batch_zeros["cyclic_mask"] = torch.zeros(1, n, dtype=torch.bool)
        out_false = self._relpos(batch_false)
        out_zeros = self._relpos(batch_zeros)
        assert torch.equal(out_false, out_zeros)
