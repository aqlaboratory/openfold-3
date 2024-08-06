import unittest

import torch
import torch.nn.functional as F

from openfold3.core.loss.distogram import all_atom_distogram_loss
from openfold3.model_implementations.af3_all_atom.config.base_config import config


class TestDistogramLoss(unittest.TestCase):
    def setup_features(self):
        # Example: UNK UNK UNK ALA GLY/A A DT
        # NumAtoms: 1 1 1 5 4 22 21
        token_mask = torch.ones((1, 10))
        restype = F.one_hot(
            torch.Tensor([[20, 20, 20, 0, 7, 7, 7, 7, 21, 28]]).long(), num_classes=32
        ).float()
        num_atoms_per_token = torch.Tensor([[1, 1, 1, 5, 1, 1, 1, 1, 22, 21]])
        start_atom_index = torch.Tensor([[0, 1, 2, 3, 8, 9, 10, 11, 12, 34]])
        asym_id = torch.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 3])

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

    def test_distogram_loss(self):
        batch = self.setup_features()
        batch_size, n_token = batch["token_mask"].shape
        no_bins = config.loss.distogram.no_bins

        logits = torch.randn((batch_size, n_token, n_token, no_bins))

        l = all_atom_distogram_loss(batch=batch, logits=logits, **config.loss.distogram)

        self.assertTrue(l.shape == ())


if __name__ == "__main__":
    unittest.main()
