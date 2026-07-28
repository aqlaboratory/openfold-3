import biotite.structure as struc
import numpy as np
import pytest


@pytest.fixture
def _make_atom_array():
    """Factory fixture for a minimal single-chain, single-residue-type AtomArray.

    Returns a callable rather than an AtomArray directly, since each test needs a
    different atom count / annotations. Shared across the data test modules that need
    a cheap, hand-built AtomArray instead of a parsed structure.
    """

    def _factory(
        n_atoms,
        sym_ids=None,
        entity_id=None,
        molecule_type_id=None,
        is_atomized=None,
        is_cyclic=None,
        bonds=None,
    ):
        aa = struc.AtomArray(n_atoms)
        aa.chain_id[:] = "A"
        aa.res_id[:] = np.arange(1, n_atoms + 1)
        aa.ins_code[:] = ""
        aa.res_name[:] = "ALA"
        aa.atom_name[:] = "CA"
        aa.element[:] = "C"
        aa.coord[:] = np.random.randn(n_atoms, 3)
        aa.occupancy = np.random.rand(n_atoms)
        aa.set_annotation("token_id", np.arange(n_atoms))
        if sym_ids is not None:
            aa.set_annotation("sym_id", np.array(sym_ids))
        if entity_id is not None:
            aa.set_annotation("entity_id", np.full(n_atoms, entity_id))
        if molecule_type_id is not None:
            aa.set_annotation("molecule_type_id", np.full(n_atoms, molecule_type_id))
        if is_atomized is not None:
            aa.set_annotation("is_atomized", np.asarray(is_atomized))
        if is_cyclic is not None:
            aa.set_annotation("is_cyclic", np.asarray(is_cyclic))
        if bonds is not None:
            aa.bonds = bonds
        return aa

    return _factory
