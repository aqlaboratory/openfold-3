import unittest

import torch
import torch.nn.functional as F

from openfold3.core.loss.confidence import (
    confidence_loss,
    pae_loss,
    pde_loss,
    plddt_loss,
    resolved_loss,
)
from openfold3.model_implementations.af3_all_atom.config import config


class TestConfidenceLoss(unittest.TestCase):
    def setup_features(self):
        # Example: UNK UNK UNK ALA GLY/A A DT
        # NumAtoms: 1 1 1 5 4 22 21
        token_mask = torch.ones((1, 10))
        restype = F.one_hot(
            torch.Tensor([[20, 20, 20, 0, 7, 7, 7, 7, 21, 28]]).long(), num_classes=32
        ).float()
        num_atoms_per_token = torch.Tensor([[1, 1, 1, 5, 1, 1, 1, 1, 22, 21]])
        start_atom_index = torch.Tensor([[0, 1, 2, 3, 8, 9, 10, 11, 12, 34]])
        asym_id = torch.Tensor([[0, 0, 0, 1, 1, 1, 1, 1, 2, 3]])

        is_protein = torch.Tensor([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]])
        is_rna = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        is_dna = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        is_atomized = torch.Tensor([[1, 1, 1, 0, 1, 1, 1, 1, 0, 0]])

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
            "is_atomized": is_atomized,
            "gt_atom_mask": gt_atom_mask,
            "gt_atom_positions": gt_atom_positions,
        }

    def test_plddt_loss(self):
        no_bins = 50
        eps = 1e-8
        bin_min = 0
        bin_max = 1

        batch = self.setup_features()
        batch_size, n_atom = batch["gt_atom_mask"].shape

        x = torch.randn_like(batch["gt_atom_positions"])

        p_b = torch.concat(
            [
                torch.zeros((batch_size, n_atom, no_bins - 1)),
                torch.ones((batch_size, n_atom, 1)),
            ],
            dim=-1,
        )

        l_plddt = plddt_loss(
            batch=batch,
            x=x,
            p_b=p_b,
            no_bins=no_bins,
            bin_min=bin_min,
            bin_max=bin_max,
            eps=eps,
        )

        self.assertTrue(l_plddt.shape == (batch_size,))

    def test_pae_loss(self):
        angle_threshold = 25
        no_bins = 64
        bin_min = 0
        bin_max = 32
        eps = 1e-8
        inf = 1e10

        batch = self.setup_features()
        batch_size, n_token = batch["token_mask"].shape

        x = torch.randn_like(batch["gt_atom_positions"])

        p_b = torch.concat(
            [
                torch.ones((batch_size, n_token, n_token, 1)),
                torch.zeros((batch_size, n_token, n_token, no_bins - 1)),
            ],
            dim=-1,
        )

        l_pae = pae_loss(
            batch=batch,
            x=x,
            p_b=p_b,
            angle_threshold=angle_threshold,
            no_bins=no_bins,
            bin_min=bin_min,
            bin_max=bin_max,
            eps=eps,
            inf=inf,
        )

        self.assertTrue(l_pae.shape == (batch_size,))

    def test_pde_loss(self):
        no_bins = 64
        bin_min = 0
        bin_max = 32
        eps = 1e-8

        batch = self.setup_features()
        batch_size, n_token = batch["token_mask"].shape

        x = torch.randn_like(batch["gt_atom_positions"])

        p_b = torch.concat(
            [
                torch.ones((batch_size, n_token, n_token, 1)),
                torch.zeros((batch_size, n_token, n_token, no_bins - 1)),
            ],
            dim=-1,
        )

        l_pde = pde_loss(
            batch=batch,
            x=x,
            p_b=p_b,
            no_bins=no_bins,
            bin_min=bin_min,
            bin_max=bin_max,
            eps=eps,
        )

        self.assertTrue(l_pde.shape == (batch_size,))

    def test_resolved_loss(self):
        no_bins = 2
        eps = 1e-8

        batch = self.setup_features()
        batch_size, n_atom = batch["gt_atom_mask"].shape

        p_b = torch.ones((batch_size, n_atom, no_bins)) * 0.5

        l_resolved = resolved_loss(batch=batch, p_b=p_b, no_bins=no_bins, eps=eps)

        self.assertTrue(l_resolved.shape == (batch_size,))

    def test_confidence_loss(self):
        batch = self.setup_features()
        batch_size, n_token = batch["token_mask"].shape
        n_atom = batch["gt_atom_mask"].shape[1]

        no_bins_plddt = config.loss.confidence.no_bins_plddt
        no_bins_pae = config.loss.confidence.no_bins_pae
        no_bins_pde = config.loss.confidence.no_bins_pde
        no_bins_resolved = config.loss.confidence.no_bins_resolved

        output = {
            "x_pred": torch.randn_like(batch["gt_atom_positions"]),
            "plddt": torch.concat(
                [
                    torch.zeros((batch_size, n_atom, no_bins_plddt - 1)),
                    torch.ones((batch_size, n_atom, 1)),
                ],
                dim=-1,
            ),
            "pae": torch.concat(
                [
                    torch.ones((batch_size, n_token, n_token, 1)),
                    torch.zeros((batch_size, n_token, n_token, no_bins_pae - 1)),
                ],
                dim=-1,
            ),
            "pde": torch.concat(
                [
                    torch.ones((batch_size, n_token, n_token, 1)),
                    torch.zeros((batch_size, n_token, n_token, no_bins_pde - 1)),
                ],
                dim=-1,
            ),
            "resolved": torch.ones((batch_size, n_atom, no_bins_resolved)) * 0.5,
        }

        l_confidence = confidence_loss(
            batch=batch, output=output, **config.loss.confidence
        )

        self.assertTrue(l_confidence.shape == ())


if __name__ == "__main__":
    unittest.main()
