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

from openfold3.core.utils.relpos import (
    apply_cyclic_offsets,
    cyclic_offset,
    relpos_complex,
)


def _linear_offsets(pos):
    """The offsets apply_cyclic_offsets is handed: pos[i] - pos[j]."""
    return pos[..., None] - pos[..., None, :]


def _diagonal(n, on_diagonal, off_diagonal):
    """[n, n] matrix of off_diagonal with on_diagonal on its diagonal."""
    return torch.full((n, n), off_diagonal).fill_diagonal_(on_diagonal)


def _assert_one_hot(t):
    """Assert that a tensor is one-hot encoded along the last dimension."""
    # hide from pytest traceback
    __tracebackhide__ = True
    assert ((t == 0) | (t == 1)).all(), "Tensor contains values other than 0 and 1"
    if t.ndim > 1:
        assert (t.sum(dim=-1) == 1).all(), (
            "One-hot encoding does not have exactly one 1 per row"
        )


def _chain_layout(chain_lengths, cyclic_flags):
    """Build (pos, asym_id, cyclic_mask) for consecutive chains of given lengths."""
    asym_id = torch.tensor(
        [i + 1 for i, length in enumerate(chain_lengths) for _ in range(length)]
    )
    cyclic_mask = torch.tensor(
        [
            flag
            for flag, length in zip(cyclic_flags, chain_lengths, strict=True)
            for _ in range(length)
        ],
        dtype=torch.bool,
    )
    pos = torch.arange(sum(chain_lengths))
    return pos.unsqueeze(0), asym_id.unsqueeze(0), cyclic_mask.unsqueeze(0)


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

    def test_simple_example(self):
        # The example shown in the `cyclic_offset` docstring

        off = cyclic_offset(torch.arange(6))
        expected = torch.tensor(
            [
                [0, -1, -2, -3, 2, 1],
                [1, 0, -1, -2, -3, 2],
                [2, 1, 0, -1, -2, -3],
                [3, 2, 1, 0, -1, -2],
                [-2, 3, 2, 1, 0, -1],
                [-1, -2, 3, 2, 1, 0],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(off, expected)

    @pytest.mark.parametrize("n", range(1, 17))
    def test_antisymmetry(self, n):
        off = cyclic_offset(torch.arange(n))
        assert torch.equal(off, -off.T)

    @pytest.mark.parametrize("n", range(2, 17, 2))
    def test_antipode_keeps_the_sign_of_the_linear_offset(self, n):
        # In an even ring, residues n // 2 apart are that far apart either way
        # round. To preserve antisymmetry, they keep the sign of the linear
        # offset.
        idx = torch.arange(n)
        linear = idx[:, None] - idx[None, :]
        antipodal = linear.abs() == n // 2

        off = cyclic_offset(idx)

        assert antipodal.sum() == n
        assert torch.equal(off[antipodal], linear[antipodal].to(off.dtype))

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_returns_int32_on_input_device(self, dtype):
        idx = torch.arange(7, dtype=dtype)
        off = cyclic_offset(idx)
        assert off.dtype == torch.int32
        assert off.device == idx.device

    @pytest.mark.parametrize("n", [2, 10, 16])
    def test_max_offset_of_even_ring(self, n):
        # For an even-length chain the maximum offset magnitude is peptide_length // 2.
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        assert int(off.abs().max()) == n // 2

    @pytest.mark.parametrize("n", [3, 5, 7, 11])
    def test_max_offset_of_odd_ring(self, n):
        # For an odd-length chain all entries are <= (n-1)//2 away from 0.
        idx = torch.arange(n)
        off = cyclic_offset(idx)
        assert int(off.abs().max()) == (n - 1) // 2

    @pytest.mark.parametrize("n", range(1, 17))
    def test_congruent_to_linear(self, n):
        idx = torch.arange(n)
        linear = idx[:, None] - idx[None, :]

        off = cyclic_offset(idx)

        assert torch.equal((off - linear) % n, torch.zeros_like(linear))

    @pytest.mark.parametrize("n", range(1, 17))
    def test_shortest_way_around_the_ring(self, n):
        idx = torch.arange(n)
        linear = idx[:, None] - idx[None, :]
        forward = linear % n
        shortest = torch.minimum(forward, n - forward)

        off = cyclic_offset(idx)

        assert torch.equal(off.abs(), shortest.to(off.dtype))

    # In a ring of 2 tail is also the antipode of head, so it keeps the sign of
    # the linear offset and off[0, 1] == -1.
    @pytest.mark.parametrize("n", range(3, 17))
    def test_head_and_tail_are_adjacent(self, n):
        off = cyclic_offset(torch.arange(n))
        assert off[0, n - 1] == 1
        assert off[n - 1, 0] == -1

    def test_uses_residue_index_values(self):
        # Inference residue indices start at 1; only differences matter.
        assert torch.equal(
            cyclic_offset(torch.arange(1, 8)), cyclic_offset(torch.arange(7))
        )

    @pytest.mark.parametrize(
        "residue_index",
        [torch.tensor([0, 1, 1, 1, 2, 3, 4]), torch.tensor([0, 1, 1, 1, 1, 1, 2, 3])],
    )
    def test_tokens_of_one_residue_are_zero_apart(self, residue_index):
        # An atomized residue contributes several tokens with the same residue index;
        # the ring still has one position per residue, so the tokens get the
        # offsets of their residues.
        n_residues = int(residue_index.max()) + 1
        per_residue = cyclic_offset(torch.arange(n_residues))

        off = cyclic_offset(residue_index)

        assert torch.equal(off, per_residue[residue_index][:, residue_index])
        assert torch.equal(off, -off.T)

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


class TestApplyCyclicOffsets:
    def test_none_mask_is_a_no_op(self):
        # The feature is optional in the batch; absent means every chain is linear.
        pos, asym_id, _ = _chain_layout([6], [False])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, None, asym_id)

        assert torch.equal(out, offset)

    def test_all_linear_mask_is_a_no_op(self):
        pos, asym_id, cyclic_mask = _chain_layout([6], [False])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)

        assert torch.equal(out, offset)

    def test_does_not_mutate_the_input(self):
        # relpos_complex reuses the linear offsets across three encodings, so wrapping
        # must not write through to the caller's tensor.
        pos, asym_id, cyclic_mask = _chain_layout([6], [True])
        offset = _linear_offsets(pos)
        before = offset.clone()

        apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)

        assert torch.equal(offset, before)

    def test_single_cyclic_chain_wraps(self):
        pos, asym_id, cyclic_mask = _chain_layout([6], [True])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)

        expected = cyclic_offset(pos.squeeze(0)).to(dtype=offset.dtype)
        assert torch.equal(out.squeeze(0), expected)
        assert out.shape == offset.shape
        assert out.dtype == offset.dtype

    def test_linear_chain_and_cross_chain_blocks_untouched(self):
        n_lin, n_cyc = 4, 6
        pos, asym_id, cyclic_mask = _chain_layout([n_lin, n_cyc], [False, True])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)[0]
        linear = offset[0]

        assert torch.equal(out[:n_lin, :n_lin], linear[:n_lin, :n_lin])
        assert torch.equal(out[:n_lin, n_lin:], linear[:n_lin, n_lin:])
        assert torch.equal(out[n_lin:, :n_lin], linear[n_lin:, :n_lin])
        assert not torch.equal(out[n_lin:, n_lin:], linear[n_lin:, n_lin:])

    def test_each_cyclic_chain_wraps_over_its_own_length(self):
        # Two cyclic chains of different lengths must each wrap at their own midpoint,
        # not at the midpoint of the complex.
        len_a, len_b = 4, 6
        pos, asym_id, cyclic_mask = _chain_layout([len_a, len_b], [True, True])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)[0]

        expected_a = cyclic_offset(pos[0, :len_a]).to(dtype=offset.dtype)
        expected_b = cyclic_offset(pos[0, len_a:]).to(dtype=offset.dtype)
        assert torch.equal(out[:len_a, :len_a], expected_a)
        assert torch.equal(out[len_a:, len_a:], expected_b)

    def test_wraps_each_sample_of_a_batch_independently(self):
        # Regression: the token indices used to be taken from the first sample's mask
        # and reused for the whole batch, which zeroed the offsets of any later sample
        # whose cyclicity differed.
        n = 6
        pos = torch.arange(n).unsqueeze(0).repeat(2, 1)
        asym_id = torch.ones(2, n, dtype=torch.int32)
        cyclic_mask = torch.tensor([[False] * n, [True] * n])
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)

        assert torch.equal(out[0], offset[0])
        assert torch.equal(out[1], cyclic_offset(pos[1]).to(dtype=offset.dtype))

    def test_supports_extra_leading_dims(self):
        # Regression: features arriving as [B, S, N] rather than [B, N] used to
        # collapse to row indices and silently produce all-zero offsets.
        n = 6
        pos = torch.arange(n).reshape(1, 1, n)
        asym_id = torch.ones(1, 1, n, dtype=torch.int32)
        cyclic_mask = torch.ones(1, 1, n, dtype=torch.bool)
        offset = _linear_offsets(pos)

        out = apply_cyclic_offsets(offset, pos, cyclic_mask, asym_id)

        assert out.shape == offset.shape
        assert torch.equal(out[0, 0], cyclic_offset(pos.reshape(-1)).to(offset.dtype))


class TestRelposComplex:
    MAX_IDX = 32
    MAX_CHAIN = 2
    # The last bin of rel_pos/rel_token and of rel_chain is for pairs on
    # different chains/residues and different entities, respectively
    POS_SENTINEL = 2 * MAX_IDX + 1
    CHAIN_SENTINEL = 2 * MAX_CHAIN + 1

    def _relpos(self, batch):
        return relpos_complex(batch, self.MAX_IDX, self.MAX_CHAIN)

    @pytest.mark.parametrize("is_cyclic", [True, False])
    def test_relpos_shape(self, is_cyclic):
        n = 10
        batch = _make_batch(n, [1] * n, [is_cyclic] * n)
        out = self._relpos(batch)
        expected_last = (2 * self.MAX_IDX + 2) * 2 + 1 + (2 * self.MAX_CHAIN + 2)
        assert out.shape == (1, n, n, expected_last)

    def _bins(self, out):
        """Decode relpos_complex output into a [*, N_token, N_token] tensor per encoding.

        Each one-hot encoding becomes the index of its hot bin, and same_entity its
        0/1 flag. Fails unless every encoding holds only 0s and 1s and every
        one-hot encoding has exactly one 1 per token pair.

        This basically undoes the one-hot encoding coming out of relpos_complex,
        making it easier to assert on while validating it.
        """
        idx_bins = 2 * self.MAX_IDX + 2
        chain_bins = 2 * self.MAX_CHAIN + 2
        # relpos_complex returns
        # torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1), so the last dim holds, in order:
        #     rel_pos      residue offset, idx_bins
        #     rel_token    token offset within a residue, idx_bins
        #     same_entity  1 flag
        #     rel_chain    chain-copy (sym_id) offset, chain_bins
        # torch.split raises if the widths don't add up to the output width.
        rel_pos, rel_token, same_entity, rel_chain = torch.split(
            out, (idx_bins, idx_bins, 1, chain_bins), dim=-1
        )
        _assert_one_hot(rel_pos)
        _assert_one_hot(rel_token)
        _assert_one_hot(rel_chain)
        assert ((same_entity == 0) | (same_entity == 1)).all()
        return {
            "rel_pos": rel_pos.argmax(dim=-1),
            "rel_token": rel_token.argmax(dim=-1),
            "same_entity": same_entity[..., 0],
            "rel_chain": rel_chain.argmax(dim=-1),
        }

    def _assert_bins(self, out, expected):
        """Assert that every encoding of relpos_complex output has the expected bins."""
        bins = self._bins(out)
        assert bins.keys() == expected.keys()
        for name, expected_bins in expected.items():
            assert torch.equal(bins[name], expected_bins), name

    def test_cyclic_wraps_only_the_residue_offset(self):
        n = 10
        batch = _make_batch(n, [1] * n, [True] * n)
        residue_index = batch["residue_index"][0]

        out = self._relpos(batch)

        self._assert_bins(
            out,
            {
                "rel_pos": (cyclic_offset(residue_index) + self.MAX_IDX).unsqueeze(0),
                # Distinct residues, so only self pairs share a residue
                "rel_token": _diagonal(n, self.MAX_IDX, self.POS_SENTINEL).unsqueeze(0),
                # Only one entity, so all pairs share the same entity
                "same_entity": torch.ones(1, n, n),
                # One chain, one sym_id: no chain-copy offset
                "rel_chain": torch.full((1, n, n), self.MAX_CHAIN),
            },
        )

    def test_cyclic_chain_with_atomized_residue(self):
        # Residue 2 is atomized into five tokens, which are 0 apart in the
        # residue offset; the ring has three residues. A previous bug cyclized
        # the token offset. The input is constructed with the second residue
        # spanning more than half of the chain's tokens to guard against that.
        residue_index = torch.tensor([1, 2, 2, 2, 2, 2, 3])
        n = len(residue_index)
        batch = _make_batch(n, [1] * n, [True] * n)
        batch["residue_index"] = residue_index.unsqueeze(0)

        out = self._relpos(batch)

        residue_offset = torch.empty((n, n), dtype=torch.int32)
        residue_offset[0] = torch.tensor([0, -1, -1, -1, -1, -1, 1])
        residue_offset[1:6] = torch.tensor([1, 0, 0, 0, 0, 0, -1])
        residue_offset[6] = torch.tensor([-1, 1, 1, 1, 1, 1, 0])

        # Token offsets only within a residue: residue 2's five tokens, and each
        # of the other residues' single token with itself
        rel_token = torch.full((n, n), self.POS_SENTINEL)
        rel_token[0, 0] = self.MAX_IDX
        rel_token[6, 6] = self.MAX_IDX
        rel_token[1:6, 1:6] = self.MAX_IDX + torch.tensor(
            [
                [0, -1, -2, -3, -4],
                [1, 0, -1, -2, -3],
                [2, 1, 0, -1, -2],
                [3, 2, 1, 0, -1],
                [4, 3, 2, 1, 0],
            ]
        )
        self._assert_bins(
            out,
            {
                "rel_pos": (residue_offset + self.MAX_IDX).unsqueeze(0),
                "rel_token": rel_token.unsqueeze(0),
                "same_entity": torch.ones(1, n, n),
                "rel_chain": torch.full((1, n, n), self.MAX_CHAIN),
            },
        )

    def test_multimer_with_one_cyclic_chain(self):
        # Only chain 2 is cyclic, so chain 1 keeps its linear residue offsets.
        n_lin = 6
        n_cyc = 6
        n = n_lin + n_cyc
        asym_ids = [1] * n_lin + [2] * n_cyc
        cyclic_mask = [False] * n_lin + [True] * n_cyc
        batch = _make_batch(n, asym_ids, cyclic_mask)
        residue_index = batch["residue_index"][0]
        asym_id = batch["asym_id"][0]

        out = self._relpos(batch)

        lin, cyc = slice(0, n_lin), slice(n_lin, n)
        rel_pos = torch.full((n, n), self.POS_SENTINEL)
        rel_pos[lin, lin] = _linear_offsets(residue_index[lin]) + self.MAX_IDX
        rel_pos[cyc, cyc] = cyclic_offset(residue_index[cyc]) + self.MAX_IDX
        # Each chain is its own entity here
        same_chain = asym_id[:, None] == asym_id[None, :]
        self._assert_bins(
            out,
            {
                "rel_pos": rel_pos.unsqueeze(0),
                "rel_token": _diagonal(n, self.MAX_IDX, self.POS_SENTINEL).unsqueeze(0),
                "same_entity": same_chain.unsqueeze(0),
                "rel_chain": torch.where(
                    same_chain, self.MAX_CHAIN, self.CHAIN_SENTINEL
                ).unsqueeze(0),
            },
        )

    @pytest.mark.parametrize("cyclic_a", [True, False])
    @pytest.mark.parametrize("cyclic_b", [True, False])
    def test_cross_chain_pairs_unaffected_by_cyclic(self, cyclic_a, cyclic_b):
        # Cross-chain token pairs should get the "different chain" sentinel
        # regardless of whether either chain is cyclic.
        n_a = 5
        n_b = 5
        n = n_a + n_b
        asym_ids = [1] * n_a + [2] * n_b
        cyclic_mask = [cyclic_a] * n_a + [cyclic_b] * n_b

        batch = _make_batch(n, asym_ids, cyclic_mask)
        bins = self._bins(self._relpos(batch))

        # The chains are different entities here
        for rows, cols in [
            (slice(0, n_a), slice(n_a, n)),
            (slice(n_a, n), slice(0, n_a)),
        ]:
            assert (bins["rel_pos"][..., rows, cols] == self.POS_SENTINEL).all()
            assert (bins["rel_token"][..., rows, cols] == self.POS_SENTINEL).all()
            assert (bins["same_entity"][..., rows, cols] == 0).all()
            assert (bins["rel_chain"][..., rows, cols] == self.CHAIN_SENTINEL).all()

    def test_all_false_cyclic_mask_is_linear(self):
        n = 8
        batch_false = _make_batch(n, [1] * n, [False] * n)
        out_false = self._relpos(batch_false)
        residue_index = batch_false["residue_index"][0]
        self._assert_bins(
            out_false,
            {
                "rel_pos": (_linear_offsets(residue_index) + self.MAX_IDX).unsqueeze(0),
                "rel_token": _diagonal(n, self.MAX_IDX, self.POS_SENTINEL).unsqueeze(0),
                "same_entity": torch.ones(1, n, n),
                "rel_chain": torch.full((1, n, n), self.MAX_CHAIN),
            },
        )

    def test_cyclic_mask_none_matches_all_false(self):
        # The feature is optional; absent means every chain is linear.
        n = 8
        batch_false = _make_batch(n, [1] * n, [False] * n)
        batch_none = _make_batch(n, [1] * n, [False] * n)
        batch_none.pop("cyclic_mask")  # Remove cyclic_mask from batch_none
        out_false = self._relpos(batch_false)
        out_none = self._relpos(batch_none)
        assert torch.equal(out_false, out_none)
