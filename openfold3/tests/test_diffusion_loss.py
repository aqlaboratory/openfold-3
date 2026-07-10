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
import torch.nn.functional as F
from torch.testing import assert_close

from openfold3.core.loss.diffusion import (
    bond_loss,
    diffusion_loss,
    mse_loss,
    smooth_lddt_loss,
    weighted_rigid_align,
)
from openfold3.core.model.structure.diffusion_module import centre_random_augmentation
from openfold3.tests.config import consts


@pytest.mark.usefixtures("seeded_rng")
class TestDiffusionLoss(unittest.TestCase):
    def setup_features(self):
        # Example: UNK UNK UNK ALA GLY/A A DT
        # NumAtoms: 1 1 1 5 4 22 21
        token_mask = torch.ones((1, 10))
        restype = F.one_hot(
            torch.Tensor([[20, 20, 20, 0, 7, 7, 7, 7, 21, 29]]).long(), num_classes=32
        ).float()
        num_atoms_per_token = torch.Tensor([[1, 1, 1, 5, 1, 1, 1, 1, 22, 21]])
        start_atom_index = torch.Tensor([[0, 1, 2, 3, 8, 9, 10, 11, 12, 34]])
        asym_id = torch.Tensor([[0, 0, 0, 1, 1, 1, 1, 1, 2, 3]])

        is_protein = torch.Tensor([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]])
        is_rna = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        is_dna = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        is_ligand = torch.Tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
        is_atomized = torch.Tensor([[1, 1, 1, 0, 1, 1, 1, 1, 0, 0]])

        token_bonds = torch.ones((1, 10, 10))

        gt_atom_mask = torch.ones((1, 55))
        gt_atom_positions = torch.randn((1, 55, 3))

        return {
            "token_mask": token_mask,
            "restype": restype,
            "num_atoms_per_token": num_atoms_per_token,
            "start_atom_index": start_atom_index,
            "asym_id": asym_id,
            "is_protein": is_protein,
            "is_rna": is_rna,
            "is_dna": is_dna,
            "is_ligand": is_ligand,
            "is_atomized": is_atomized,
            "token_bonds": token_bonds,
            "ground_truth": {
                "atom_resolved_mask": gt_atom_mask,
                "atom_positions": gt_atom_positions,
            },
            "loss_weights": {
                "bond": torch.Tensor([1.0]),
                "smooth_lddt": torch.Tensor([1.0]),
                "mse": torch.Tensor([1.0]),
            },
        }

    def test_weighted_rigid_align(self):
        batch_size = consts.batch_size
        n_atom = 2 * consts.n_res

        x_gt = torch.randn((batch_size, n_atom, 3))
        w = torch.concat(
            [
                torch.ones((batch_size, consts.n_res)),
                torch.ones((batch_size, consts.n_res)) * 5,
            ],
            dim=-1,
        )
        atom_mask_gt = torch.ones((batch_size, n_atom))

        x = centre_random_augmentation(x_gt, atom_mask_gt)
        x_align = weighted_rigid_align(
            x=x, x_gt=x_gt, w=w, atom_mask_gt=atom_mask_gt, eps=consts.eps
        )

        self.assertTrue(x_align.shape == (batch_size, n_atom, 3))
        self.assertTrue(torch.sum(torch.abs(x_align - x_gt) > 1e-5) == 0)

    def test_weighted_rigid_align_ignores_masked_atoms(self):
        # Masked-out atoms must not enter the centroid/covariance: with garbage
        # at the masked positions the valid atoms must still recover x_gt.
        k, g = 12, 6
        theta = torch.tensor(0.7)
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        x_gt_valid = torch.randn(1, k, 3)
        x_valid = x_gt_valid @ rot.T + torch.tensor([3.0, -1.0, 2.0])

        x_gt = torch.cat([x_gt_valid, 1e3 * torch.randn(1, g, 3)], dim=1)
        x = torch.cat([x_valid, 1e3 * torch.randn(1, g, 3)], dim=1)
        mask = torch.cat([torch.ones(1, k), torch.zeros(1, g)], dim=1)

        x_align = weighted_rigid_align(
            x=x, x_gt=x_gt, w=torch.ones(1, k + g), atom_mask_gt=mask, eps=1e-8
        )

        assert_close(x_align[:, :k], x_gt_valid)

    def test_mse_loss(self):
        n_sample = 2
        dna_weight = 5
        rna_weight = 5
        ligand_weight = 10

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        all_ones_mask = torch.ones_like(batch["is_protein"])

        mse_unmasked = mse_loss(
            x=x,
            batch=batch,
            loss_token_mask=all_ones_mask,
            dna_weight=dna_weight,
            rna_weight=rna_weight,
            ligand_weight=ligand_weight,
            eps=consts.eps,
        )

        self.assertTrue(mse_unmasked.shape == (batch_size, n_sample))
        self.assertTrue((mse_unmasked < 1e-5).all())

        # Check when token mask has some zeros
        mostly_zeros_mask = torch.zeros_like(batch["is_protein"])
        mostly_zeros_mask[:, :2] = 1

        mse_masked = mse_loss(
            x=x,
            batch=batch,
            loss_token_mask=mostly_zeros_mask,
            dna_weight=dna_weight,
            rna_weight=rna_weight,
            ligand_weight=ligand_weight,
            eps=consts.eps,
        )
        assert torch.equal(torch.nonzero(mse_masked), torch.nonzero(mostly_zeros_mask))
        assert torch.all(torch.not_equal(mse_masked, mse_unmasked))

    def test_bond_loss(self):
        n_sample = 2

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )

        loss = bond_loss(x=x, batch=batch, eps=consts.eps)

        self.assertTrue(loss.shape == (batch_size, n_sample))
        self.assertTrue((loss < 1e-5).all())

    def test_bond_loss_value(self):
        # Two coincident clusters of m single-atom polymer+ligand tokens make
        # the bond mask all-true. Predicted cluster B is twice as far as the
        # ground-truth B, so every cross-cluster pair has squared error exactly
        # 1 while within-cluster pairs and the diagonal are 0; half the ordered
        # pairs are cross-cluster, so the masked mean is exactly 0.5. The input
        # is large enough that a less precise accumulator for the mask sum (e.g.
        # bf16) would have a noticeable difference.
        m = 175
        n = 2 * m
        x_gt = torch.zeros((1, n, 3))
        x_gt[:, m:, 0] = 1.0
        x = torch.zeros((1, n, 3))
        x[:, m:, 0] = 2.0
        batch = {
            "token_mask": torch.ones((1, n)),
            "num_atoms_per_token": torch.ones((1, n)),
            "is_protein": torch.ones((1, n)),
            "is_dna": torch.zeros((1, n)),
            "is_rna": torch.zeros((1, n)),
            "is_ligand": torch.ones((1, n)),
            "token_bonds": torch.ones((1, n, n)),
            "ground_truth": {
                "atom_resolved_mask": torch.ones((1, n)),
                "atom_positions": x_gt,
            },
        }

        loss = bond_loss(x=x, batch=batch, eps=1e-8)

        self.assertAlmostEqual(loss.item(), 0.5)

    def test_smooth_lddt_loss(self):
        n_sample = 2

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )

        all_ones_mask = torch.ones_like(batch["is_protein"])
        loss = smooth_lddt_loss(
            x=x, batch=batch, loss_token_mask=all_ones_mask, eps=1e-8
        )

        gt_loss = 1 - 0.25 * (
            torch.sigmoid(torch.Tensor([0.5]))
            + torch.sigmoid(torch.Tensor([1.0]))
            + torch.sigmoid(torch.Tensor([2.0]))
            + torch.sigmoid(torch.Tensor([4.0]))
        )
        gt_loss = gt_loss[None, ...]

        self.assertTrue(loss.shape == (batch_size, n_sample))
        self.assertTrue((torch.abs(loss - gt_loss) < 1e-5).all())

        # Check when token mask has some zeros
        mostly_zeros_mask = torch.zeros_like(batch["is_protein"])
        mostly_zeros_mask[:, :2] = 1
        loss_masked = smooth_lddt_loss(
            x=x, batch=batch, loss_token_mask=mostly_zeros_mask, eps=1e-8
        )

        assert torch.equal(torch.nonzero(loss_masked), torch.nonzero(mostly_zeros_mask))
        assert torch.all(torch.not_equal(loss_masked, loss))

    def test_smooth_lddt_loss_ignores_masked_atoms(self):
        # Masked-out atoms must not affect the loss: adding extra atoms whose
        # ground truth would count but whose prediction is wildly wrong (would
        # drag the loss up markedly if they leaked), masked out, must give the
        # same loss as not having them at all.
        n = 10
        extra = 5
        x_gt = torch.randn(1, n + extra, 3)
        x = torch.randn(1, n + extra, 3)
        mask = torch.cat([torch.ones(1, n), torch.zeros(1, extra)], dim=1)

        def run(positions_gt, positions, resolved_mask):
            n_tok = positions_gt.shape[1]
            batch = {
                "token_mask": torch.ones((1, n_tok)),
                "num_atoms_per_token": torch.ones((1, n_tok)),
                "is_dna": torch.zeros((1, n_tok)),
                "is_rna": torch.zeros((1, n_tok)),
                "ground_truth": {
                    "atom_resolved_mask": resolved_mask,
                    "atom_positions": positions_gt,
                },
            }
            return smooth_lddt_loss(
                x=positions,
                batch=batch,
                loss_token_mask=torch.ones((1, n_tok)),
                eps=1e-8,
            )

        loss_valid = run(x_gt[:, :n, :], x[:, :n, :], mask[:, :n])
        loss_masked = run(x_gt, x, mask)

        self.assertAlmostEqual(loss_masked.item(), loss_valid.item(), delta=1e-6)

    def _test_diffusion_loss(self, batch):
        n_sample = 2
        sigma_data = 16

        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )

        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size))

        loss, _ = diffusion_loss(batch=batch, x=x, t=t, sigma_data=sigma_data)

        gt_loss = 1 - 0.25 * (
            torch.sigmoid(torch.Tensor([0.5]))
            + torch.sigmoid(torch.Tensor([1.0]))
            + torch.sigmoid(torch.Tensor([2.0]))
            + torch.sigmoid(torch.Tensor([4.0]))
        )

        self.assertTrue(loss.shape == ())
        self.assertTrue((torch.abs(loss - gt_loss) < 1e-5).all())

    def test_diffusion_loss_without_disordered_flag(self):
        batch = self.setup_features()
        self._test_diffusion_loss(batch)

    def test_diffusion_loss_with_disordered_flag(self):
        batch = self.setup_features()
        batch["loss_weights"]["disable_non_protein_diffusion_weights"] = torch.tensor(
            [True], dtype=torch.bool
        )
        self._test_diffusion_loss(batch)


if __name__ == "__main__":
    unittest.main()
