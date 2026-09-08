from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from openfold3.core.data.primitives.featurization.structure import create_token_bonds
from openfold3.core.data.primitives.structure.query import (
    add_query_covalent_bonds,
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)


def make_query(endpoint=None, reverse=False, ligand=None):
    chains = [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "GCG"}]
    if ligand is not None:
        chains.append({"molecule_type": "ligand", "chain_ids": "X", **ligand})
    if reverse:
        chains.reverse()
    return Query.model_validate(
        {
            "chains": chains,
            "covalent_bonds": [
                {"atom1": ["A", 2, "SG"], "atom2": endpoint or ["B", 2, "SG"]}
            ],
        }
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "ligand, endpoint",
    [
        (None, ["B", 2, "SG"]),
        ({"ccd_codes": ["EOH"]}, ["X", 1, "C1"]),
        ({"smiles": "CCO"}, ["X", 1, "C1"]),
    ],
)
def test_named_bonds_reach_token_features(reverse, ligand, endpoint):
    query = make_query(endpoint, reverse, ligand)
    atoms = structure_with_ref_mols_from_query(query).atom_array
    indices = [
        int(
            np.flatnonzero(
                (atoms.chain_id == atom.chain_id)
                & (atoms.res_id == atom.residue_id)
                & (atoms.atom_name == atom.atom_name)
            )[0]
        )
        for atom in query.covalent_bonds[0]
    ]
    tokenize_atom_array(atoms)
    assert atoms.is_atomized[indices].all()
    tokens = np.unique(atoms.token_id)
    positions = [int(np.flatnonzero(tokens == atoms.token_id[i])[0]) for i in indices]
    bonds = create_token_bonds(atoms, tokens)
    assert bonds[positions[0], positions[1]] == 1
    assert bonds[positions[1], positions[0]] == 1


@pytest.mark.parametrize(
    "endpoint", [["Z", 2, "SG"], ["B", 99, "SG"], ["B", 2, "TYPO"]]
)
def test_missing_endpoint_fails(endpoint):
    with pytest.raises(ValueError, match="expected one atom, found 0"):
        structure_with_ref_mols_from_query(make_query(endpoint))


def test_self_bond_fails():
    with pytest.raises(ValueError, match="itself"):
        structure_with_ref_mols_from_query(make_query(["A", 2, "SG"]))


def test_duplicates_preserve_existing_bonds():
    query = make_query()
    atoms = structure_with_ref_mols_from_query(query).atom_array
    original = atoms.bonds.as_array().copy()
    query.covalent_bonds.append(
        query.covalent_bonds[0]._replace(
            atom1=query.covalent_bonds[0].atom2, atom2=query.covalent_bonds[0].atom1
        )
    )
    add_query_covalent_bonds(atoms, query)
    np.testing.assert_array_equal(atoms.bonds.as_array(), original)


@pytest.mark.parametrize("bonds", [None, []])
def test_disabled_is_noop(bonds):
    query = make_query()
    query.covalent_bonds = bonds
    atoms = structure_with_ref_mols_from_query(query).atom_array
    original_bonds = atoms.bonds
    original = atoms.copy()
    add_query_covalent_bonds(atoms, query)
    assert atoms.bonds is original_bonds
    assert atoms == original


@pytest.mark.parametrize("endpoint", [["A", 0, "SG"], ["A", 2, 0], ["A", 2, ""]])
def test_invalid_selector_schema(endpoint):
    with pytest.raises(ValidationError):
        make_query(endpoint)


def test_example_builds():
    root = Path(__file__).resolve().parents[6]
    queries = InferenceQuerySet.from_json(
        root / "examples/example_inference_inputs/covalent_bonds.json"
    )
    structure_with_ref_mols_from_query(queries.queries["disulfide"])
