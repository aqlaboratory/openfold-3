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

import tests.compare_utils as compare_utils
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
        atom_mask = torch.ones((batch_size, n_atom))

        x = centre_random_augmentation(x_gt, atom_mask)
        x_align = weighted_rigid_align(x=x, x_gt=x_gt, w=w, atom_mask=atom_mask)

        self.assertTrue(x_align.shape == (batch_size, n_atom, 3))
        self.assertTrue(torch.sum(torch.abs(x_align - x_gt) > 1e-5) == 0)

    def test_mse_loss(self):
        batch_size = consts.batch_size
        n_token = 4 * consts.n_res
        n_atom = 16 * consts.n_res
        alpha_dna = 5
        alpha_rna = 5
        alpha_ligand = 10

        x1_gt = torch.randn((batch_size, n_atom, 3))
        x2_gt = torch.randn((batch_size, n_atom, 3))
        atom_mask = torch.ones((batch_size, n_atom)).bool()
        atom_mask[1, -44:] = 0

        x1 = centre_random_augmentation(x1_gt, atom_mask)
        x2 = centre_random_augmentation(x2_gt, atom_mask)

        batch = {
            "is_polymer": torch.concat(
                [
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_dna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_rna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_ligand": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
        }

        mse = mse_loss(
            batch=batch,
            x=x1,
            x_gt=x2,
            atom_mask=atom_mask,
            alpha_dna=alpha_dna,
            alpha_rna=alpha_rna,
            alpha_ligand=alpha_ligand,
        )
        mse_gt = mse_loss(
            batch=batch,
            x=x1_gt,
            x_gt=x2_gt,
            atom_mask=atom_mask,
            alpha_dna=alpha_dna,
            alpha_rna=alpha_rna,
            alpha_ligand=alpha_ligand,
        )

        self.assertTrue(mse.shape == (batch_size,))
        self.assertTrue(torch.sum((mse - mse_gt) > 1e-5) == 0)

    @compare_utils.skip_unless_cuda_available()
    def test_bond_loss(self):
        batch_size = consts.batch_size
        n_token = 4 * consts.n_res
        n_atom = 16 * consts.n_res

        x_gt = torch.randn((batch_size, n_atom, 3))
        atom_mask = torch.ones((batch_size, n_atom))
        atom_mask[1, -44:] = 0

        x = centre_random_augmentation(x_gt, atom_mask)

        batch = {
            "is_polymer": torch.concat(
                [
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_dna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_rna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_ligand": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
            "token_bonds": torch.ones((batch_size, n_token, n_token)),
        }

        loss = bond_loss(batch, x, x_gt, atom_mask)

        self.assertTrue(loss.shape == (batch_size,))
        self.assertTrue(torch.sum(loss > 1e-5) == 0)

    def test_smooth_lddt_loss(self):
        batch_size = consts.batch_size
        n_token = 4 * consts.n_res
        n_atom = 16 * consts.n_res

        x_gt = torch.randn((batch_size, n_atom, 3))
        atom_mask = torch.ones((batch_size, n_atom))
        atom_mask[1, -44:] = 0

        x = centre_random_augmentation(x_gt, atom_mask)

        batch = {
            "is_polymer": torch.concat(
                [
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_dna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_rna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_ligand": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
        }

        loss = smooth_lddt_loss(batch, x, x_gt, atom_mask)

        self.assertTrue(loss.shape == (batch_size,))

    @compare_utils.skip_unless_cuda_available()
    def test_diffusion_loss(self):
        batch_size = consts.batch_size
        n_token = 4 * consts.n_res
        n_atom = 16 * consts.n_res
        sigma_data = 16
        alpha_bond = 1

        x_gt = torch.randn((batch_size, n_atom, 3))
        atom_mask = torch.ones((batch_size, n_atom))
        atom_mask[1, -44:] = 0

        x = centre_random_augmentation(x_gt, atom_mask)

        batch = {
            "is_polymer": torch.concat(
                [
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_dna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_rna": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "is_ligand": torch.concat(
                [
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.zeros((batch_size, consts.n_res)),
                    torch.ones((batch_size, consts.n_res)),
                ],
                dim=-1,
            ),
            "atom_to_token_index": torch.eye(n_token)
            .repeat_interleave(4, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
            "token_bonds": torch.ones((batch_size, n_token, n_token)),
        }

        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size))

        loss = diffusion_loss(
            batch=batch,
            x=x,
            x_gt=x_gt,
            atom_mask=atom_mask,
            t=t,
            sigma_data=sigma_data,
            alpha_bond=alpha_bond,
        )

        self.assertTrue(loss.shape == ())


if __name__ == "__main__":
    unittest.main()
