import unittest

import torch
import torch.nn.functional as F

from openfold3.core.loss.confidence import (
    all_atom_experimentally_resolved_loss,
    all_atom_plddt_loss,
    confidence_loss,
    pae_loss,
    pde_loss,
)
from openfold3.projects.af3_all_atom.config.base_config import (
    project_config as af3_project_config,
)


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
            "ground_truth": {
                "atom_resolved_mask": gt_atom_mask,
                "atom_positions": gt_atom_positions,
            },
            "loss_weights": {
                "bond": torch.Tensor([1.0]),
                "smooth_lddt": torch.Tensor([4.0]),
                "mse": torch.Tensor([4.0]),
                "plddt": torch.Tensor([1e-4]),
                "pde": torch.Tensor([0.0]),
                "experimentally_resolved": torch.Tensor([0.2]),
                "pae": torch.Tensor([1.0]),
                "distogram": torch.Tensor([3e-2]),
            },
        }

    def test_plddt_loss(self):
        no_bins = 50
        eps = 1e-8
        bin_min = 0
        bin_max = 1

        batch = self.setup_features()
        batch_size, n_atom = batch["ground_truth"]["atom_resolved_mask"].shape

        x = torch.randn_like(batch["ground_truth"]["atom_positions"])

        logits = torch.randn((batch_size, n_atom, no_bins))

        l_plddt = all_atom_plddt_loss(
            batch=batch,
            x=x,
            logits=logits,
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

        x = torch.randn_like(batch["ground_truth"]["atom_positions"])

        logits = torch.randn((batch_size, n_token, n_token, no_bins))

        l_pae = pae_loss(
            batch=batch,
            x=x,
            logits=logits,
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

        x = torch.randn_like(batch["ground_truth"]["atom_positions"])

        logits = torch.randn((batch_size, n_token, n_token, no_bins))

        l_pde = pde_loss(
            batch=batch,
            x=x,
            logits=logits,
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
        batch_size, n_atom = batch["ground_truth"]["atom_resolved_mask"].shape

        logits = torch.randn((batch_size, n_atom, no_bins))

        l_resolved = all_atom_experimentally_resolved_loss(
            batch=batch, logits=logits, no_bins=no_bins, eps=eps
        )

        self.assertTrue(l_resolved.shape == (batch_size,))

    def test_confidence_loss(self):
        batch = self.setup_features()
        batch_size, n_token = batch["token_mask"].shape
        n_atom = batch["ground_truth"]["atom_resolved_mask"].shape[1]

        config = af3_project_config.model

        no_bins_plddt = config.architecture.loss_module.confidence.plddt.no_bins
        no_bins_pae = config.architecture.loss_module.confidence.pae.no_bins
        no_bins_pde = config.architecture.loss_module.confidence.pde.no_bins
        no_bins_resolved = (
            config.architecture.loss_module.confidence.experimentally_resolved.no_bins
        )

        output = {
            "atom_positions_predicted": torch.randn_like(
                batch["ground_truth"]["atom_positions"]
            ),
            "plddt_logits": torch.randn((batch_size, n_atom, no_bins_plddt)),
            "pae_logits": torch.randn((batch_size, n_token, n_token, no_bins_pae)),
            "pde_logits": torch.randn((batch_size, n_token, n_token, no_bins_pde)),
            "experimentally_resolved_logits": torch.randn(
                (batch_size, n_atom, no_bins_resolved)
            ),
        }

        l_confidence, _ = confidence_loss(
            batch=batch, output=output, **config.architecture.loss_module.confidence
        )

        self.assertTrue(l_confidence.shape == ())


if __name__ == "__main__":
    unittest.main()
