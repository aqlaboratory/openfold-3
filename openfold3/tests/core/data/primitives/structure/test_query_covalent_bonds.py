from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
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

EXAMPLES_DIR = Path(__file__).resolve().parents[6] / "examples/example_inference_inputs"


def make_query(endpoint=None, ligand=None):
    chains = [{"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "GCG"}]
    if ligand is not None:
        chains.append({"molecule_type": "ligand", "chain_ids": "X", **ligand})
    return Query.model_validate(
        {
            "chains": chains,
            "covalent_bonds": [
                {"atom1": ["A", 2, "SG"], "atom2": endpoint or ["B", 2, "SG"]}
            ],
        }
    )


@pytest.mark.parametrize(
    "chain_order", [(0, 1), (1, 0)], ids=["protein-first", "ligand-first"]
)
@pytest.mark.parametrize(
    "ligand, endpoint",
    [
        (None, ["B", 2, "SG"]),
        ({"ccd_codes": ["EOH"]}, ["X", 1, "C1"]),
        ({"smiles": "CCO"}, ["X", 1, "C1"]),
    ],
)
def test_named_bonds_reach_token_features(chain_order, ligand, endpoint):
    query = make_query(endpoint, ligand)
    query.chains = [query.chains[i] for i in chain_order if i < len(query.chains)]
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
    queries = InferenceQuerySet.from_json(EXAMPLES_DIR / "covalent_bonds.json")
    structure_with_ref_mols_from_query(queries.queries["disulfide"])


@pytest.mark.parametrize(
    "chain_order, glycan_ref_index",
    [
        pytest.param((0, 1), -1, id="protein-first"),
        pytest.param((1, 0), 0, id="glycan-first"),
    ],
)
def test_asn_linked_two_sugar_glycan(chain_order, glycan_ref_index):
    query = InferenceQuerySet.from_json(
        EXAMPLES_DIR / "query_asn_two_sugar_glycan.json"
    ).queries["asn_two_sugar_glycan"]
    query.chains = [query.chains[i] for i in chain_order]
    structure = structure_with_ref_mols_from_query(query)
    atoms = structure.atom_array
    # Reference molecules follow builder residue order, including the three
    # protein residues; select the glycan independently of query chain order.
    glycan_ref = structure.processed_reference_mols[glycan_ref_index]
    mol = glycan_ref.mol
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) == 2
    assert all(len(ring) == 6 for ring in rings)
    names = [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]
    assert set(atoms.atom_name[atoms.chain_id == "G"]) == set(names)
    assert glycan_ref.in_crop_mask.all()

    def atom_index(chain, name):
        indices = np.flatnonzero(
            (atoms.chain_id == chain) & (atoms.res_id == 1) & (atoms.atom_name == name)
        )
        assert len(indices) == 1
        return int(indices[0])

    nd2 = atom_index("A", "ND2")
    attachment = atom_index("G", "C1")
    ring_atoms = set(rings[0]) | set(rings[1])
    # Audit the internal glycosidic oxygen from the processed reference graph,
    # rather than relying on SMILES atom order or an assumed oxygen name.
    bridges = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "O"
        and atom.GetIdx() not in ring_atoms
        and len(atom.GetNeighbors()) == 2
        and any(n.GetIdx() in rings[0] for n in atom.GetNeighbors())
        and any(n.GetIdx() in rings[1] for n in atom.GetNeighbors())
    ]
    assert len(bridges) == 1
    bridge = bridges[0]
    pairs = [(nd2, attachment)] + [
        (atom_index("G", names[bridge.GetIdx()]), atom_index("G", names[n.GetIdx()]))
        for n in bridge.GetNeighbors()
    ]
    structure_pairs = {tuple(sorted(pair)) for pair in atoms.bonds.as_array()[:, :2]}
    assert all(tuple(sorted(pair)) in structure_pairs for pair in pairs)

    tokenize_atom_array(atoms)
    assert atoms.is_atomized[(atoms.chain_id == "A") & (atoms.res_id == 1)].all()
    assert atoms.is_atomized[atoms.chain_id == "G"].all()
    tokens = np.unique(atoms.token_id)
    token_positions = {int(token): i for i, token in enumerate(tokens)}
    features = featurize_structure_of3(
        atoms, len(tokens), is_gt=False, add_perm_features=False
    )
    batch = openfold_batch_collator([features])
    # Collation adds the leading sample axis to the token adjacency matrix.
    assert batch["token_bonds"].shape == (1, len(tokens), len(tokens))
    bonds = batch["token_bonds"][0]
    for first, second in pairs:
        i = token_positions[int(atoms.token_id[first])]
        j = token_positions[int(atoms.token_id[second])]
        assert bonds[i, j] == bonds[j, i] == 1
