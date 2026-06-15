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

from openfold3.core.loss.diffusion import (
    bond_loss,
    diffusion_loss,
    mse_loss,
    smooth_lddt_loss,
    smooth_lddt_loss_ball_query,
    weighted_rigid_align,
)
from openfold3.core.model.structure.diffusion_module import centre_random_augmentation
from openfold3.tests.config import consts


class TestDiffusionLoss(unittest.TestCase):
    def to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {key: self.to_device(value, device) for key, value in obj.items()}
        return obj

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

    def test_smooth_lddt_ball_query_matches_dense(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        n_sample = 2
        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        batch["ground_truth"]["atom_resolved_mask"][:, 0] = 0
        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        top_k = x.shape[-2] - 1

        all_ones_mask = torch.ones_like(batch["is_protein"])
        dense = smooth_lddt_loss(
            x=x, batch=batch, loss_token_mask=all_ones_mask, eps=consts.eps
        )
        ball_query = smooth_lddt_loss_ball_query(
            x=x,
            batch=batch,
            loss_token_mask=all_ones_mask,
            eps=consts.eps,
            top_k=top_k,
        )
        torch.testing.assert_close(ball_query, dense, atol=1e-5, rtol=1e-5)

        mostly_zeros_mask = torch.zeros_like(batch["is_protein"])
        mostly_zeros_mask[:, :3] = 1
        dense_masked = smooth_lddt_loss(
            x=x, batch=batch, loss_token_mask=mostly_zeros_mask, eps=consts.eps
        )
        ball_query_masked = smooth_lddt_loss_ball_query(
            x=x,
            batch=batch,
            loss_token_mask=mostly_zeros_mask,
            eps=consts.eps,
            top_k=top_k,
        )
        torch.testing.assert_close(
            ball_query_masked, dense_masked, atol=1e-5, rtol=1e-5
        )

    def test_smooth_lddt_ball_query_requires_top_k(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        n_sample = 2
        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        all_ones_mask = torch.ones_like(batch["is_protein"])

        with pytest.raises(ValueError, match="smooth_lddt_top_k"):
            smooth_lddt_loss_ball_query(
                x=x,
                batch=batch,
                loss_token_mask=all_ones_mask,
                eps=consts.eps,
                top_k=None,
            )

    def test_smooth_lddt_ball_query_gradient_matches_dense(self):
        # Ball-query backward uses atomicAdd in the CUDA kernel; tolerance is
        # slightly looser than the forward-only matches_dense test.
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        n_sample = 2
        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        batch["ground_truth"]["atom_resolved_mask"][:, 0] = 0
        x_base = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        all_ones_mask = torch.ones_like(batch["is_protein"])
        top_k = x_base.shape[-2] - 1

        x_dense = x_base.detach().clone().requires_grad_(True)
        dense = smooth_lddt_loss(
            x=x_dense, batch=batch, loss_token_mask=all_ones_mask, eps=consts.eps
        )
        dense.sum().backward()

        x_bq = x_base.detach().clone().requires_grad_(True)
        bq = smooth_lddt_loss_ball_query(
            x=x_bq,
            batch=batch,
            loss_token_mask=all_ones_mask,
            eps=consts.eps,
            top_k=top_k,
        )
        bq.sum().backward()

        torch.testing.assert_close(bq, dense, atol=1e-5, rtol=1e-5)
        # Backward kernel uses atomicAdd; reduction order also differs from
        # the dense [N,N] path. Tolerance is loosened accordingly.
        torch.testing.assert_close(x_bq.grad, x_dense.grad, atol=1e-3, rtol=5e-3)

    def test_smooth_lddt_ball_query_backward_runs_subsampled(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        x = batch["ground_truth"]["atom_positions"].clone()
        x = x + 0.1 * torch.randn_like(x)
        x.requires_grad_(True)
        all_ones_mask = torch.ones_like(batch["is_protein"])

        # top_k well below n_atom: reservoir sampling kicks in
        loss = smooth_lddt_loss_ball_query(
            x=x,
            batch=batch,
            loss_token_mask=all_ones_mask,
            eps=consts.eps,
            top_k=16,
            seed=7,
        )
        loss.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    def test_smooth_lddt_ball_query_bf16_path(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        all_ones_mask = torch.ones_like(batch["is_protein"])
        x = batch["ground_truth"]["atom_positions"].clone()
        x = x + 0.05 * torch.randn_like(x)

        x_fp = x.float().detach().clone().requires_grad_(True)
        loss_fp = smooth_lddt_loss_ball_query(
            x=x_fp,
            batch=batch,
            loss_token_mask=all_ones_mask,
            eps=consts.eps,
            top_k=x.shape[-2] - 1,
            seed=0,
        )
        loss_fp.sum().backward()

        x_bf = x.to(torch.bfloat16).detach().clone().requires_grad_(True)
        loss_bf = smooth_lddt_loss_ball_query(
            x=x_bf,
            batch=batch,
            loss_token_mask=all_ones_mask,
            eps=consts.eps,
            top_k=x.shape[-2] - 1,
            seed=0,
        )
        loss_bf.sum().backward()

        assert x_bf.grad.dtype == torch.bfloat16
        # Forward and gradient should be close between bf16 and fp32 paths.
        torch.testing.assert_close(
            loss_bf.float(), loss_fp.float(), atol=5e-3, rtol=5e-3
        )
        torch.testing.assert_close(x_bf.grad.float(), x_fp.grad, atol=5e-2, rtol=5e-2)

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

    def test_diffusion_loss_ball_query_matches_dense(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        n_sample = 2
        sigma_data = 16
        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]
        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size, device=device))
        top_k = x.shape[-2] - 1

        dense_loss, dense_breakdown = diffusion_loss(
            batch=batch,
            x=x,
            t=t,
            sigma_data=sigma_data,
            smooth_lddt_backend="dense",
        )
        ball_query_loss, ball_query_breakdown = diffusion_loss(
            batch=batch,
            x=x,
            t=t,
            sigma_data=sigma_data,
            smooth_lddt_backend="ball_query",
            smooth_lddt_top_k=top_k,
        )

        torch.testing.assert_close(ball_query_loss, dense_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            ball_query_breakdown["smooth_lddt_loss"],
            dense_breakdown["smooth_lddt_loss"],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_diffusion_loss_ball_query_rejects_chunking(self):
        if not torch.cuda.is_available():
            pytest.skip("Ball-query smooth lDDT requires CUDA")

        n_sample = 2
        sigma_data = 16
        device = torch.device("cuda")

        batch = self.to_device(self.setup_features(), device)
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]
        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size, device=device))
        top_k = x.shape[-2] - 1

        with pytest.raises(ValueError, match="chunked loss execution"):
            diffusion_loss(
                batch=batch,
                x=x,
                t=t,
                sigma_data=sigma_data,
                chunk_size=1,
                smooth_lddt_backend="ball_query",
                smooth_lddt_top_k=top_k,
            )

    def test_diffusion_loss_rejects_unknown_smooth_lddt_backend(self):
        n_sample = 2
        sigma_data = 16

        batch = self.setup_features()
        batch_size = batch["ground_truth"]["atom_resolved_mask"].shape[0]
        x = centre_random_augmentation(
            xl=batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1)),
            atom_mask=batch["ground_truth"]["atom_resolved_mask"],
        )
        t = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size))

        with pytest.raises(ValueError, match="smooth_lddt_backend"):
            diffusion_loss(
                batch=batch,
                x=x,
                t=t,
                sigma_data=sigma_data,
                smooth_lddt_backend="unknown",
            )


if __name__ == "__main__":
    unittest.main()
