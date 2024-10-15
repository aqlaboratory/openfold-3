# Copyright 2021 AlQuraishi Laboratory
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
import torch.nn.functional as F

from openfold3.core.loss.diffusion import (
    bond_loss,
    diffusion_loss,
    mse_loss,
    smooth_lddt_loss,
    weighted_rigid_align,
)
from openfold3.core.model.structure.diffusion_module import centre_random_augmentation
from tests.config import consts


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

        mse = mse_loss(
            x=x,
            batch=batch,
            dna_weight=dna_weight,
            rna_weight=rna_weight,
            ligand_weight=ligand_weight,
            eps=consts.eps,
        )

        self.assertTrue(mse.shape == (batch_size, n_sample))
        self.assertTrue((mse < 1e-5).all())

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

    def test_smooth_lddt_loss(self):
        n_sample = 2

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )

        loss = smooth_lddt_loss(x=x, batch=batch, eps=consts.eps)

        gt_loss = 1 - 0.25 * (
            torch.sigmoid(torch.Tensor([0.5]))
            + torch.sigmoid(torch.Tensor([1.0]))
            + torch.sigmoid(torch.Tensor([2.0]))
            + torch.sigmoid(torch.Tensor([4.0]))
        )
        gt_loss = gt_loss[None, ...]

        self.assertTrue(loss.shape == (batch_size, n_sample))
        self.assertTrue((torch.abs(loss - gt_loss) < 1e-5).all())

    def test_diffusion_loss(self):
        n_sample = 2
        sigma_data = 16

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]

        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )

        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size))

        loss, _ = diffusion_loss(
            batch=batch,
            x=x,
            t=t,
            sigma_data=sigma_data,
            eps=consts.eps,
        )

        gt_loss = 1 - 0.25 * (
            torch.sigmoid(torch.Tensor([0.5]))
            + torch.sigmoid(torch.Tensor([1.0]))
            + torch.sigmoid(torch.Tensor([2.0]))
            + torch.sigmoid(torch.Tensor([4.0]))
        )

        self.assertTrue(loss.shape == ())
        self.assertTrue((torch.abs(loss - gt_loss) < 1e-5).all())


if __name__ == "__main__":
    unittest.main()
