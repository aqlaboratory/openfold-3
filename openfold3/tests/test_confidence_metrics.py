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
from biotite import structure

from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
from openfold3.core.metrics.confidence import (
    get_bin_centers,
    logits_to_expected_error,
    probs_to_expected_error,
)
from openfold3.core.metrics.sample_ranking import (
    compute_has_clash,
    compute_ptm,
)
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.data_utils import random_of3_features


class TestConfidenceMetrics(unittest.TestCase):
    def test_get_bin_centers_basic(self):
        # 0..1 split into 5 bins
        centers = get_bin_centers(0.0, 1.0, 5, device="cpu", dtype=torch.float32)
        expected = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertTrue(torch.allclose(centers, expected, atol=1e-6))

    def test_probs_to_expected_error_uniform(self):
        # Uniform distribution across bins should give the midpoint (0.5) for [0,1]
        no_bins = 10
        probs = torch.ones(no_bins) / no_bins
        actual_error = torch.tensor(0.5)
        expected_error = probs_to_expected_error(probs, 0.0, 1.0, no_bins)
        self.assertTrue(torch.allclose(expected_error, actual_error))

    def test_logits_to_expected_error_matches_probability_path(self):
        logits = torch.randn(2, 3, 7)
        expected = probs_to_expected_error(
            torch.softmax(logits, dim=-1), bin_min=0.0, bin_max=1.0, no_bins=7
        )
        actual = logits_to_expected_error(
            logits, bin_min=0.0, bin_max=1.0, no_bins=7
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_logits_to_expected_error_can_return_probs(self):
        logits = torch.randn(2, 3, 7)
        expected_probs = torch.softmax(logits, dim=-1)
        expected = probs_to_expected_error(
            expected_probs, bin_min=0.0, bin_max=1.0, no_bins=7
        )
        actual, actual_probs = logits_to_expected_error(
            logits,
            bin_min=0.0,
            bin_max=1.0,
            no_bins=7,
            return_probs=True,
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual_probs, expected_probs))

    def test_confidence_scores_match_with_cpu_offloaded_aux_outputs(self):
        batch_size = 1
        num_samples = 1
        n_token = 6
        n_msa = 3
        n_templ = 1
        config = OF3ProjectEntry().get_model_config_with_presets()
        config.architecture.heads.pae.enabled = True
        config.confidence.pde.return_probs = False
        config.confidence.pae.return_probs = False
        config.confidence.distogram.return_contact_probs = False

        batch = random_of3_features(batch_size, n_token, n_msa, n_templ)
        batch["is_protein"].zero_()
        n_atom = batch["atom_mask"].shape[-1]
        batch["atom_array"] = [structure.AtomArray(n_atom)]

        outputs = {
            "atom_positions_predicted": torch.randn(batch_size, num_samples, n_atom, 3),
            "plddt_logits": torch.randn(batch_size, num_samples, n_atom, 50),
            "pde_logits": torch.randn(
                batch_size,
                num_samples,
                n_token,
                n_token,
                config.architecture.heads.pde.c_out,
            ),
            "distogram_logits": torch.randn(
                batch_size,
                n_token,
                n_token,
                config.architecture.heads.distogram.c_out,
            ),
            "pae_logits": torch.randn(
                batch_size,
                num_samples,
                n_token,
                n_token,
                config.architecture.heads.pae.c_out,
            ),
        }
        expected = get_confidence_scores(batch=batch, outputs=outputs, config=config)
        actual = get_confidence_scores(
            batch=batch,
            outputs={k: v.cpu() for k, v in outputs.items()},
            config=config,
        )

        def assert_close_tree(actual_value, expected_value):
            if isinstance(expected_value, dict):
                self.assertEqual(actual_value.keys(), expected_value.keys())
                for key in expected_value:
                    assert_close_tree(actual_value[key], expected_value[key])
            else:
                self.assertTrue(torch.allclose(actual_value, expected_value))

        assert_close_tree(actual, expected)

    def test_shape(self):
        batch_size, num_samples, n_atom = 1, 5, 16
        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        logits = torch.randn(
            (batch_size, num_samples, n_atom, config.confidence.plddt.no_bins)
        )
        probs = torch.softmax(logits, dim=-1)
        error = probs_to_expected_error(probs, **config.confidence.plddt)
        self.assertTrue(error.shape == (batch_size, num_samples, n_atom))

    def test_compute_has_clash_basic_abs_and_per_sample(self):
        """
        Two chains (non-contiguous asym ids).
        Sample 0: many close inter-chain pairs -> clash=1.
        Sample 1: far apart -> clash=0.
        """
        num_samples = 2
        asym_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)  # [N_atom]
        is_polymer = torch.ones_like(asym_id, dtype=torch.bool)  # all polymer
        atom_mask = torch.ones_like(asym_id, dtype=torch.bool)

        # positions: [S, N, 3]
        pos = torch.zeros(num_samples, asym_id.numel(), 3, dtype=torch.float32)

        # Sample 0: chain 20 atoms close (< threshold) to first two atoms of chain 10
        # place chain 10 atoms
        pos[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        pos[0, 1] = torch.tensor([5.0, 0.0, 0.0])
        pos[0, 2] = torch.tensor([10.0, 0.0, 0.0])
        pos[0, 3] = torch.tensor([0.5, 0.0, 0.0])  # close to atom 0
        pos[0, 4] = torch.tensor([5.5, 0.0, 0.0])  # close to atom 1
        pos[0, 5] = torch.tensor([50.0, 0.0, 0.0])  # far

        # Sample 1: move chain 20 far away (no clashes)
        pos[1, 0] = torch.tensor([0.0, 0.0, 0.0])
        pos[1, 1] = torch.tensor([5.0, 0.0, 0.0])
        pos[1, 2] = torch.tensor([10.0, 0.0, 0.0])
        pos[1, 3] = torch.tensor([100.0, 0.0, 0.0])
        pos[1, 4] = torch.tensor([105.0, 0.0, 0.0])
        pos[1, 5] = torch.tensor([110.0, 0.0, 0.0])

        has_clash = compute_has_clash(
            asym_id=asym_id,
            atom_positions_predicted=pos,
            atom_mask=atom_mask,
            is_polymer=is_polymer,
            threshold=1.1,
            violation_abs=1,
            violation_frac=0.5,
        )
        self.assertTrue(torch.allclose(has_clash, torch.tensor([1.0, 0.0])))

        has_clash = compute_has_clash(
            asym_id=asym_id,
            atom_positions_predicted=pos,
            atom_mask=atom_mask,
            is_polymer=is_polymer,
            threshold=1.1,
            violation_abs=2,
            violation_frac=0.5,
        )
        self.assertTrue(torch.allclose(has_clash, torch.tensor([1.0, 0.0])))

    def test_compute_has_clash_ignores_nonpolymer(self):
        """
        Inter-chain close contacts from a non-polymer chain should not trigger clashes.
        """
        asym_id = torch.tensor([3, 3, 7, 7], dtype=torch.long)  # two chains
        # mark chain 7 as non-polymer
        is_polymer = torch.tensor([True, True, False, False], dtype=torch.bool)
        atom_mask = torch.ones(4, dtype=torch.bool)

        pos = torch.zeros(1, 4, 3, dtype=torch.float32)
        # put chain 3 and chain 7 close — but chain 7 is non-polymer
        pos[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        pos[0, 1] = torch.tensor([2.0, 0.0, 0.0])
        pos[0, 2] = torch.tensor([0.5, 0.0, 0.0])  # non-polymer
        pos[0, 3] = torch.tensor([2.5, 0.0, 0.0])  # non-polymer

        has_clash = compute_has_clash(
            asym_id=asym_id,
            atom_positions_predicted=pos,
            atom_mask=atom_mask,
            is_polymer=is_polymer,
            threshold=1.1,
            violation_abs=1,
            violation_frac=0.5,
        )
        # Expect no clash because only polymer-polymer chain pairs should be checked
        self.assertTrue(torch.allclose(has_clash, torch.tensor([0.0])))

    def _make_two_bin_weights(self, N_d: int, bin_min=0.0, bin_max=2.0, no_bins=2):
        # Helper: compute the two bin weights exactly like compute_ptm
        bin_centers = get_bin_centers(
            bin_min, bin_max, no_bins, device="cpu", dtype=torch.float32
        )
        N_eff = max(N_d, 19)
        d0 = 1.24 * (N_eff - 15.0) ** (1.0 / 3.0) - 1.8
        w = 1.0 / (1.0 + (bin_centers / d0) ** 2)
        return w  # [no_bins]

    def test_compute_ptm_has_frame_exclusion_ipTM(self):
        """
        ipTM outer-max must ignore tokens with has_frame=False (per-sample).
        We craft pairs so token 1 would have the highest ipTM, then mask it out and
        expect the next-highest (token 2) to win.
        """
        S, N, B = 1, 3, 2
        # Non-contiguous chain ids: [10, 20, 10]
        asym_id = torch.tensor([10, 20, 10])
        token_mask = torch.ones(N, dtype=torch.bool)
        has_frame = torch.tensor([[True, False, True]])  # token 1 invalid

        # Build logits [S, N, N, B]; set per-(i,j) bin preference
        logits = torch.zeros(S, N, N, B)
        # We'll use bin_min=0, bin_max=2, no_bins=2  -> bin centers [0.5, 1.5]
        # Make inter-chain (1,0) and (1,2) strongly favor bin0 (high weight)
        logits[0, 1, 0, 0] = 8.0
        logits[0, 1, 0, 1] = -8.0
        logits[0, 1, 2, 0] = 8.0
        logits[0, 1, 2, 1] = -8.0
        # Make (0,1) low-weight (bin1)
        logits[0, 0, 1, 1] = 8.0
        logits[0, 0, 1, 0] = -8.0
        # Make (2,1) high-weight (bin0)
        logits[0, 2, 1, 0] = 8.0
        logits[0, 2, 1, 1] = -8.0
        # Everything else irrelevant

        # Compute ipTM over all tokens (mask_i = all True)
        iptm = compute_ptm(
            logits=logits,
            has_frame=has_frame,
            bin_min=0.0,
            bin_max=2.0,
            no_bins=2,
            mask_i=token_mask,
            asym_id=asym_id,
            interface=True,
        )  # [S]

        # Expected: token 1 would be max, but it's masked out -> token 2 wins.
        w = self._make_two_bin_weights(N_d=3, bin_min=0.0, bin_max=2.0, no_bins=2)
        expected = w[0]  # bin0 (high-weight) used by (2,1)
        self.assertTrue(torch.allclose(iptm, expected.expand_as(iptm)))


if __name__ == "__main__":
    unittest.main()
