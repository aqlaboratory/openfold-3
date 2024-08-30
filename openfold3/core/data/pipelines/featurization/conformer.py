"""Conformer featurization pipeline."""

import torch


def featurize_conformers_dummy_af3(batch_size, n_atom):
    """Temporary function to generate dummy template features."""
    return {
        "ref_pos": torch.randn((batch_size, n_atom, 3)).float(),
        "ref_mask": torch.ones((batch_size, n_atom)).int(),
        "ref_element": torch.ones((batch_size, n_atom, 128)).int(),
        "ref_charge": torch.ones((batch_size, n_atom)).float(),
        "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)).int(),
        "ref_space_uid": torch.zeros((batch_size, n_atom)).int(),
    }
