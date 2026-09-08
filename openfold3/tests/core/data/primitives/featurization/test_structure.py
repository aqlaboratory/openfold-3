import numpy as np
import pytest
import torch

from openfold3.core.data.primitives.featurization.structure import (
    maybe_create_cyclic_mask,
)


class TestCreateCyclicMask:
    def test_missing_annotation_defaults_to_linear(self, _make_atom_array):
        # Structures parsed from mmCIF -- i.e. all training data -- have no is_cyclic
        # annotation. The mask must still be produced, or downstream consumers that
        # read it unconditionally (relpos_complex) blow up.
        n_atoms = 6
        atom_array = _make_atom_array(n_atoms)

        mask = maybe_create_cyclic_mask(atom_array, np.arange(n_atoms))

        assert mask.dtype == torch.bool
        assert mask.shape == (n_atoms,)
        assert not mask.any()

    @pytest.mark.parametrize("is_cyclic", [True, False])
    def test_uniform_annotation_is_propagated(self, _make_atom_array, is_cyclic):
        n_atoms = 6
        atom_array = _make_atom_array(n_atoms, is_cyclic=[is_cyclic] * n_atoms)

        mask = maybe_create_cyclic_mask(atom_array, np.arange(n_atoms))

        assert mask.dtype == torch.bool
        assert mask.tolist() == [is_cyclic] * n_atoms

    def test_mask_is_read_at_token_starts(self, _make_atom_array):
        # The annotation is per-atom but the feature is per-token, so only the first
        # atom of each token is sampled. Vary cyclicity per atom and check that the
        # sampled -- not the raw -- values come back.
        n_atoms = 6
        per_atom = [True, False, False, True, False, False]
        atom_array = _make_atom_array(n_atoms, is_cyclic=per_atom)
        token_starts = np.array([0, 3])

        mask = maybe_create_cyclic_mask(atom_array, token_starts)

        assert mask.tolist() == [True, True]

    def test_non_bool_annotation_is_coerced(self, _make_atom_array):
        # biotite stores annotations as whatever dtype it is handed; an int-valued
        # is_cyclic must still come back as bool, since relpos does bitwise ops on it.
        n_atoms = 4
        atom_array = _make_atom_array(n_atoms, is_cyclic=np.array([1, 1, 0, 0]))

        mask = maybe_create_cyclic_mask(atom_array, np.arange(n_atoms))

        assert mask.dtype == torch.bool
        assert mask.tolist() == [True, True, False, False]

    def test_empty_token_starts(self, _make_atom_array):
        atom_array = _make_atom_array(4, is_cyclic=[True] * 4)

        mask = maybe_create_cyclic_mask(atom_array, np.array([], dtype=int))

        assert mask.shape == (0,)
        assert mask.dtype == torch.bool
