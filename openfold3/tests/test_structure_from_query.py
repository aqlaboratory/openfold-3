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

# TODO: Add more tests for general inference inputs

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure.io import pdb, pdbx
from rdkit import Chem

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.query import (
    StructureWithReferenceMolecules,
    _apply_manual_leaving_atoms,
    _materialize_query_covalent_bonds,
    infer_ccd_leaving_atoms,
    normalize_manual_leaving_atoms,
    processed_reference_molecule_from_mol,
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    get_token_count,
    tokenize_atom_array,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Atom,
    Query,
)


class _FakeCCDColumn:
    def __init__(self, values):
        self.values = values

    def as_array(self):
        return np.asarray(self.values)


def _fake_ccd_entry(atom_names, elements, leaving_flags, bonds):
    return {
        "chem_comp_atom": {
            "atom_id": _FakeCCDColumn(atom_names),
            "type_symbol": _FakeCCDColumn(elements),
            "pdbx_leaving_atom_flag": _FakeCCDColumn(leaving_flags),
        },
        "chem_comp_bond": {
            "atom_id_1": _FakeCCDColumn([bond[0] for bond in bonds]),
            "atom_id_2": _FakeCCDColumn([bond[1] for bond in bonds]),
        },
    }


def _make_atom_array(
    chain_ids,
    residue_ids,
    residue_names,
    atom_names,
    molecule_types,
    bonds=(),
):
    atom_array = struc.AtomArray(len(atom_names))
    atom_array.chain_id = chain_ids
    atom_array.res_id = residue_ids
    atom_array.res_name = residue_names
    atom_array.atom_name = atom_names
    atom_array.element = [name.rstrip("0123456789").upper() for name in atom_names]
    atom_array.set_annotation("molecule_type_id", molecule_types)
    atom_array.bonds = struc.BondList(
        len(atom_array), np.asarray(bonds, dtype=np.uint32).reshape((-1, 3))
    )
    return atom_array


# A standard peptide query
standard_peptide_query = Query.model_validate(
    {
        "query_name": "std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
            }
        ],
    }
)

# A peptide query with non-canonical residues methionine sulfoxide (MHO) and
# selenocysteine (SEC)
non_canonical_peptide_query = Query.model_validate(
    {
        "query_name": "non_std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
                "non_canonical_residues": {
                    "1": "MHO",
                    "3": "SEC",
                },
            }
        ],
    }
)


def _serialize_structure_with_ref_mols(
    swrm: StructureWithReferenceMolecules,
) -> dict[str, np.ndarray]:
    """Flatten a StructureWithReferenceMolecules into a dict of numpy arrays.

    All fields that the former assert_atomarray_equal / assert_ref_mols_equal helpers
    compared are captured here as plain numpy arrays, making the snapshot format
    independent of biotite's internal serialization protocol.
    """
    arrays: dict[str, np.ndarray] = {}

    aa = swrm.atom_array
    for annot in sorted(aa.get_annotation_categories()):
        arrays[f"aa__{annot}"] = np.asarray(getattr(aa, annot))
    arrays["aa__coord"] = aa.coord
    if aa.bonds is not None:
        arrays["aa__bonds"] = aa.bonds.as_array()

    for i, ref_mol in enumerate(swrm.processed_reference_mols):
        p = f"mol_{i}"
        arrays[f"{p}__smiles"] = np.array(
            [Chem.MolToSmiles(ref_mol.mol, canonical=False)]
        )
        arrays[f"{p}__in_crop_mask"] = ref_mol.in_crop_mask
        if ref_mol.component_id is not None:
            arrays[f"{p}__component_id"] = np.array([ref_mol.component_id])

        atom_names = [
            a.GetProp("annot_atom_name")
            for a in ref_mol.mol.GetAtoms()
            if a.HasProp("annot_atom_name")
        ]
        if atom_names:
            arrays[f"{p}__annot_atom_name"] = np.array(atom_names)

        used_masks = [
            a.GetProp("annot_used_atom_mask")
            for a in ref_mol.mol.GetAtoms()
            if a.HasProp("annot_used_atom_mask")
        ]
        if used_masks:
            arrays[f"{p}__annot_used_atom_mask"] = np.array(used_masks)

        if ref_mol.permutations is not None:
            for j, perm in enumerate(ref_mol.permutations):
                arrays[f"{p}__perm_{j}"] = perm

    return arrays


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(standard_peptide_query, id="standard_peptide"),
        pytest.param(non_canonical_peptide_query, id="non_canonical_peptide"),
    ],
)
def test_structure_from_query(query: Query, ndarrays_regression):
    """Tests that the generated structure and reference molecules matches gt."""
    structure_with_ref_mols = structure_with_ref_mols_from_query(query)
    ndarrays_regression.check(
        _serialize_structure_with_ref_mols(structure_with_ref_mols)
    )


def test_smiles_with_explicit_hydrogen():
    """Tests that SMILES with explicit hydrogens can be processed.

    Regression test for a bug where explicit hydrogens in the input molecule
    caused a length mismatch between the atom mask and the molecule after
    conformer generation (which removes hydrogens).
    """
    # SMILES with explicit hydrogen - this triggered the bug
    smiles_with_explicit_h = "[H]/C=C\\Cl"
    mol = Chem.MolFromSmiles(smiles_with_explicit_h)

    # Should not raise an error
    ref_mol = processed_reference_molecule_from_mol(mol)

    # Verify mask length matches mol atom count
    assert ref_mol.mol.GetNumAtoms() == len(ref_mol.in_crop_mask)

    # Featurization should also succeed
    features = featurize_reference_conformers_of3(
        [ref_mol],
        add_ref_space_uid_to_perm=False,
    )
    assert "ref_pos" in features


def test_smiles_ligand_cif_auth_seq_id_is_numeric(tmp_path):
    """Regression test for SMILES ligands being written with missing auth seq IDs."""
    query = Query.model_validate(
        {
            "query_name": "protein_smiles_ligand",
            "chains": [
                {
                    "molecule_type": "protein",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "chain_ids": "A",
                },
                {
                    "molecule_type": "ligand",
                    "smiles": "NCCc1cc(O)c(O)cc1",
                    "chain_ids": "X",
                },
            ],
        }
    )
    atom_array = structure_with_ref_mols_from_query(query).atom_array

    cif_path = tmp_path / "protein_smiles_ligand.cif"
    write_structure(atom_array, cif_path)

    cif_block = get_cif_block(pdbx.CIFFile.read(cif_path))
    atom_site = cif_block["atom_site"]
    ligand_mask = atom_site["label_asym_id"].as_array() == "X"

    assert ligand_mask.any()
    assert set(atom_site["label_seq_id"].as_array()[ligand_mask]) == {"."}
    assert set(atom_site["auth_seq_id"].as_array()[ligand_mask]) == {"1"}


def _smiles_ligand(chain_id, smiles, ligand_name=None, **extra):
    chain = {
        "molecule_type": "ligand",
        "chain_ids": chain_id,
        "smiles": smiles,
    }
    if ligand_name is not None:
        chain["ligand_name"] = ligand_name
    return chain | extra


def test_smiles_ligand_names_override_defaults_and_write_outputs(tmp_path):
    query = Query.model_validate(
        {
            "chains": [
                _smiles_ligand("X", "CCO", " abc123 "),
                _smiles_ligand("Y", "CCN"),
                _smiles_ligand("Z", "CCC"),
            ]
        }
    )
    atom_array = structure_with_ref_mols_from_query(query).atom_array
    expected_names = {"X": "ABC123", "Y": "LIG1", "Z": "LIG0"}

    cif_path = tmp_path / "custom_smiles_residue_names.cif"
    write_structure(atom_array, cif_path)
    cif_block = get_cif_block(pdbx.CIFFile.read(cif_path))
    atom_site = cif_block["atom_site"]
    for chain_id, residue_name in expected_names.items():
        chain_mask = atom_site["label_asym_id"].as_array() == chain_id
        assert set(atom_site["label_comp_id"].as_array()[chain_mask]) == {residue_name}
    assert set(expected_names.values()) <= set(cif_block["chem_comp"]["id"].as_array())

    pdb_path = tmp_path / "custom_smiles_residue_names.pdb"
    atom_array.b_factor[:] = 0.0
    write_structure(atom_array, pdb_path)
    pdb_array = pdb.PDBFile.read(pdb_path).get_structure(model=1)
    for chain_id, residue_name in {"X": "ABC", "Y": "LIG", "Z": "LIG"}.items():
        assert set(pdb_array.res_name[pdb_array.chain_id == chain_id]) == {residue_name}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param(
            {"ligand_name": "DR-G"},
            "ASCII letters and digits",
            id="invalid-character",
        ),
        pytest.param(
            {"ligand_name": "ß"},
            "ASCII letters and digits",
            id="unicode-case-expansion",
        ),
        pytest.param(
            {"ligand_name": "ALA"},
            "standard residue name",
            id="standard-residue",
        ),
        pytest.param(
            {
                "molecule_type": "protein",
                "sequence": "A",
                "ligand_name": "DRG",
            },
            "only be specified for a ligand",
            id="polymer",
        ),
        pytest.param(
            {"ccd_codes": "ATP", "ligand_name": "DRG"},
            "only be specified for a ligand",
            id="smiles-and-ccd",
        ),
    ],
)
def test_smiles_ligand_name_validation(overrides, match):
    chain = _smiles_ligand("X", "CCO") | overrides
    with pytest.raises(ValueError, match=match):
        Query.model_validate({"chains": [chain]})


@pytest.mark.parametrize(
    ("chains", "match"),
    [
        pytest.param(
            [
                _smiles_ligand("X", "CCO", "DRG"),
                _smiles_ligand("Y", "CCO", "INH"),
            ],
            "same SMILES string",
            id="same-smiles-different-names",
        ),
        pytest.param(
            [
                _smiles_ligand("X", "CCO", "LIG0"),
                _smiles_ligand("Y", "CCN"),
            ],
            "Distinct SMILES strings",
            id="different-smiles-same-name",
        ),
        pytest.param(
            [
                _smiles_ligand("X", "CCO", "ATP"),
                _smiles_ligand("Y", "CCN", "MHO"),
                {
                    "molecule_type": "ligand",
                    "chain_ids": "Z",
                    "ccd_codes": "ATP",
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "P",
                    "sequence": "A",
                    "non_canonical_residues": {1: "MHO"},
                },
            ],
            r"\['ATP', 'MHO'\]",
            id="non-smiles-component-names",
        ),
    ],
)
def test_smiles_ligand_name_conflicts(chains, match):
    query = Query.model_validate({"chains": chains})
    with pytest.raises(ValueError, match=match):
        structure_with_ref_mols_from_query(query)


def test_materialize_inter_chain_bond_and_mark_canonical_endpoint_residues():
    atom_array = _make_atom_array(
        chain_ids=["A", "A", "B", "B"],
        residue_ids=[1, 1, 1, 1],
        residue_names=["ALA", "ALA", "ALA", "ALA"],
        atom_names=["CA", "C", "N", "CA"],
        molecule_types=[MoleculeType.PROTEIN] * 4,
        bonds=[(0, 1, struc.BondType.SINGLE), (2, 3, struc.BondType.SINGLE)],
    )
    query = Query.model_validate(
        {
            "query_name": "cross_chain_cn",
            "chains": [],
            "covalent_bonds": [[["A", 1, "C"], ["B", 1, "N"]]],
        }
    )

    _materialize_query_covalent_bonds(atom_array, query)

    assert (1, 2, int(struc.BondType.SINGLE)) in {
        tuple(map(int, bond)) for bond in atom_array.bonds.as_array()
    }
    assert atom_array.force_atomization.tolist() == [True, True, True, True]


def test_materialize_rejects_same_chain_bond():
    atom_array = _make_atom_array(
        chain_ids=["A", "A"],
        residue_ids=[1, 2],
        residue_names=["ALA", "ALA"],
        atom_names=["C", "N"],
        molecule_types=[MoleculeType.PROTEIN] * 2,
    )
    query = Query.model_validate(
        {
            "query_name": "same_chain",
            "chains": [],
            "covalent_bonds": [[["A", 1, "C"], ["A", 2, "N"]]],
        }
    )

    with pytest.raises(ValueError, match="same_chain.*inter-chain bonds only"):
        _materialize_query_covalent_bonds(atom_array, query)

    assert atom_array.bonds.get_bond_count() == 0
    assert "force_atomization" not in atom_array.get_annotation_categories()


def test_missing_covalent_atom_reports_available_names():
    atom_array = _make_atom_array(
        chain_ids=["A", "L"],
        residue_ids=[1, 1],
        residue_names=["CYS", "LIG"],
        atom_names=["SG", "C1"],
        molecule_types=[MoleculeType.PROTEIN, MoleculeType.LIGAND],
    )
    query = Query.model_validate(
        {
            "query_name": "missing_atom",
            "chains": [],
            "covalent_bonds": [[["A", 1, "SG"], ["L", 1, "C7"]]],
        }
    )

    with pytest.raises(ValueError, match="missing_atom.*valid atom names.*C1"):
        _materialize_query_covalent_bonds(atom_array, query)


def test_invalid_custom_bond_does_not_partially_mutate_structure():
    atom_array = _make_atom_array(
        chain_ids=["A", "L"],
        residue_ids=[1, 1],
        residue_names=["CYS", "LIG"],
        atom_names=["SG", "C1"],
        molecule_types=[MoleculeType.PROTEIN, MoleculeType.LIGAND],
    )
    query = Query.model_validate(
        {
            "query_name": "atomic_validation",
            "chains": [],
            "covalent_bonds": [
                [["A", 1, "SG"], ["L", 1, "C1"]],
                [["A", 1, "SG"], ["L", 1, "missing"]],
            ],
        }
    )

    with pytest.raises(ValueError, match="covalent bond 1"):
        _materialize_query_covalent_bonds(atom_array, query)

    assert atom_array.bonds.as_array().shape == (0, 3)
    assert "force_atomization" not in atom_array.get_annotation_categories()


def test_materialize_rejects_reversed_duplicate_bond():
    atom_array = _make_atom_array(
        chain_ids=["A", "L"],
        residue_ids=[1, 1],
        residue_names=["CYS", "LIG"],
        atom_names=["SG", "C1"],
        molecule_types=[MoleculeType.PROTEIN, MoleculeType.LIGAND],
    )
    query = Query.model_validate(
        {
            "query_name": "duplicate",
            "chains": [],
            "covalent_bonds": [
                [["A", 1, "SG"], ["L", 1, "C1"]],
                [["L", 1, "C1"], ["A", 1, "SG"]],
            ],
        }
    )

    with pytest.raises(ValueError, match="duplicates an earlier custom bond"):
        _materialize_query_covalent_bonds(atom_array, query)

    assert atom_array.bonds.get_bond_count() == 0
    assert "force_atomization" not in atom_array.get_annotation_categories()


def test_materialize_rejects_endpoint_selected_as_leaving_atom():
    atom_array = _make_atom_array(
        chain_ids=["A", "L"],
        residue_ids=[1, 1],
        residue_names=["CYS", "LIG"],
        atom_names=["SG", "C1"],
        molecule_types=[MoleculeType.PROTEIN, MoleculeType.LIGAND],
    )
    query = Query.model_validate(
        {
            "query_name": "removed_endpoint",
            "chains": [],
            "covalent_bonds": [[["A", 1, "SG"], ["L", 1, "C1"]]],
            "leaving_atoms": [["L", 1, "C1"]],
        }
    )

    with pytest.raises(ValueError, match="also selected as a leaving atom"):
        _materialize_query_covalent_bonds(atom_array, query)


@pytest.mark.parametrize(
    ("intrinsic_type", "expect_error"),
    [
        (struc.BondType.SINGLE, False),
        (struc.BondType.DOUBLE, True),
    ],
)
def test_materialize_handles_existing_intrinsic_bond(intrinsic_type, expect_error):
    atom_array = _make_atom_array(
        chain_ids=["A", "L"],
        residue_ids=[1, 1],
        residue_names=["CYS", "LIG"],
        atom_names=["SG", "C1"],
        molecule_types=[MoleculeType.PROTEIN, MoleculeType.LIGAND],
        bonds=[(0, 1, intrinsic_type)],
    )
    query = Query.model_validate(
        {
            "query_name": "intrinsic",
            "chains": [],
            "covalent_bonds": [[["A", 1, "SG"], ["L", 1, "C1"]]],
        }
    )

    if expect_error:
        with pytest.raises(ValueError, match="conflicts with intrinsic bond type"):
            _materialize_query_covalent_bonds(atom_array, query)
        assert "force_atomization" not in atom_array.get_annotation_categories()
    else:
        _materialize_query_covalent_bonds(atom_array, query)
        assert atom_array.bonds.get_bond_count() == 1
        assert atom_array.force_atomization.tolist() == [True, False]


def test_manual_leaving_atom_updates_structure_and_reference_mask():
    mol = Chem.MolFromSmiles("CCl")
    for atom, atom_name in zip(mol.GetAtoms(), ["C1", "CL1"], strict=True):
        atom.SetProp("annot_atom_name", atom_name)
    processed_reference_mol = ProcessedReferenceMolecule(
        mol=mol,
        in_crop_mask=np.ones(2, dtype=bool),
    )
    atom_array = _make_atom_array(
        chain_ids=["L", "L"],
        residue_ids=[1, 1],
        residue_names=["LIG", "LIG"],
        atom_names=["C1", "CL1"],
        molecule_types=[MoleculeType.LIGAND] * 2,
        bonds=[(0, 1, struc.BondType.SINGLE)],
    )
    query = Query.model_validate(
        {
            "query_name": "manual_leaving",
            "chains": [],
            "leaving_atoms": [["L", 1, "CL1"]],
        }
    )

    atom_array = _apply_manual_leaving_atoms(
        atom_array,
        {("L", 1): [processed_reference_mol]},
        query,
    )

    assert atom_array.atom_name.tolist() == ["C1"]
    assert atom_array.bonds.get_bond_count() == 0
    assert processed_reference_mol.in_crop_mask.tolist() == [True, False]


def test_structure_from_query_applies_manual_leaving_atom_and_custom_bond():
    query = Query.model_validate(
        {
            "query_name": "public_structure_integration",
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "smiles": "CCl",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "S",
                    "smiles": "CC",
                },
            ],
            "covalent_bonds": [[["L", 1, "C1"], ["S", 1, "C1"]]],
            "leaving_atoms": [["L", 1, "CL1"]],
        }
    )

    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array

    assert not np.any((atom_array.chain_id == "L") & (atom_array.atom_name == "CL1"))
    endpoint_indices = [
        int(
            np.where(
                (atom_array.chain_id == chain_id)
                & (atom_array.res_id == 1)
                & (atom_array.atom_name == "C1")
            )[0][0]
        )
        for chain_id in ("L", "S")
    ]
    custom_pair = tuple(sorted(endpoint_indices))
    bond_pairs = {
        tuple(sorted((int(atom_1), int(atom_2))))
        for atom_1, atom_2, _ in atom_array.bonds.as_array()
    }
    assert custom_pair in bond_pairs
    assert structure.processed_reference_mols[0].in_crop_mask.tolist() == [True, False]

    tokenize_atom_array(atom_array)
    features = featurize_structure_of3(
        atom_array=atom_array,
        n_tokens=get_token_count(atom_array),
        is_gt=False,
        add_perm_features=False,
    )
    endpoint_token_ids = atom_array.token_id[endpoint_indices]
    assert features["token_bonds"][tuple(endpoint_token_ids)] == 1
    assert features["token_bonds"][tuple(endpoint_token_ids[::-1])] == 1


def test_public_protein_smiles_bond_atomizes_polymer_and_aligns_features():
    query = Query.model_validate(
        {
            "query_name": "protein_smiles_integration",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "C",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "smiles": "N",
                },
            ],
            "covalent_bonds": [[["A", 1, "SG"], ["L", 1, "N1"]]],
        }
    )

    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array
    tokenize_atom_array(atom_array)
    n_tokens = get_token_count(atom_array)
    structure_features = featurize_structure_of3(
        atom_array=atom_array,
        n_tokens=n_tokens,
        is_gt=False,
        add_perm_features=False,
    )
    reference_features = featurize_reference_conformers_of3(
        structure.processed_reference_mols,
        add_ref_space_uid_to_perm=False,
    )

    protein_mask = atom_array.chain_id == "A"
    assert np.all(atom_array.is_atomized[protein_mask])
    assert len(np.unique(atom_array.token_id[protein_mask])) == np.count_nonzero(
        protein_mask
    )
    assert int(structure_features["num_atoms_per_token"].sum()) == len(atom_array)
    assert reference_features["ref_pos"].shape[0] == len(atom_array)

    endpoint_indices = [
        int(
            np.where(
                (atom_array.chain_id == chain_id)
                & (atom_array.res_id == 1)
                & (atom_array.atom_name == atom_name)
            )[0][0]
        )
        for chain_id, atom_name in (("A", "SG"), ("L", "N1"))
    ]
    endpoint_token_ids = atom_array.token_id[endpoint_indices]
    assert structure_features["token_bonds"][tuple(endpoint_token_ids)] == 1
    assert structure_features["token_bonds"][tuple(endpoint_token_ids[::-1])] == 1


def _ccd_inference_query(manual_leaving_atoms=None):
    return Query.model_validate(
        {
            "query_name": "ccd_inference",
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "ccd_codes": "XXX",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "S",
                    "smiles": "N",
                },
            ],
            "covalent_bonds": [[["L", 1, "C1"], ["S", 1, "N1"]]],
            "leaving_atoms": manual_leaving_atoms,
        }
    )


def test_infer_ccd_leaving_atoms_is_endpoint_local_and_idempotent():
    ccd = {
        "XXX": _fake_ccd_entry(
            atom_names=["C1", "CL1", "H1", "O1"],
            elements=["C", "CL", "H", "O"],
            leaving_flags=["N", "Y", "Y", "Y"],
            bonds=[("C1", "CL1"), ("CL1", "H1")],
        )
    }

    effective_query = infer_ccd_leaving_atoms(_ccd_inference_query(), ccd)
    assert effective_query.leaving_atoms == [Atom("L", 1, "CL1")]
    assert infer_ccd_leaving_atoms(effective_query, ccd) == effective_query


def test_infer_ccd_leaving_atoms_deduplicates_manual_entries():
    ccd = {
        "XXX": _fake_ccd_entry(
            atom_names=["C1", "CL1"],
            elements=["C", "CL"],
            leaving_flags=["N", "Y"],
            bonds=[("C1", "CL1")],
        )
    }
    manual_atom = ["L", 1, "CL1"]

    effective_query = infer_ccd_leaving_atoms(
        _ccd_inference_query([manual_atom, manual_atom]), ccd
    )

    assert effective_query.leaving_atoms == [Atom("L", 1, "CL1")]


def test_normalize_manual_leaving_atoms_does_not_require_ccd_inference():
    query = _ccd_inference_query([["L", 1, "CL1"], ["L", 1, "CL1"], ["L", 1, "O1"]])

    effective_query = normalize_manual_leaving_atoms(query)

    assert effective_query.leaving_atoms == [
        Atom("L", 1, "CL1"),
        Atom("L", 1, "O1"),
    ]
    assert len(query.leaving_atoms) == 3


def test_infer_ccd_leaving_atoms_with_no_candidate_changes_nothing():
    ccd = {
        "XXX": _fake_ccd_entry(
            atom_names=["C1", "O1"],
            elements=["C", "O"],
            leaving_flags=["N", "N"],
            bonds=[("C1", "O1")],
        )
    }

    effective_query = infer_ccd_leaving_atoms(_ccd_inference_query(), ccd)
    assert effective_query.leaving_atoms is None


def test_infer_ccd_leaving_atoms_rejects_ambiguous_groups():
    ccd = {
        "XXX": _fake_ccd_entry(
            atom_names=["C1", "CL1", "CL2"],
            elements=["C", "CL", "CL"],
            leaving_flags=["N", "Y", "Y"],
            bonds=[("C1", "CL1"), ("C1", "CL2")],
        )
    }

    with pytest.raises(ValueError, match="multiple endpoint-local.*CL1.*CL2"):
        infer_ccd_leaving_atoms(_ccd_inference_query(), ccd)


@pytest.mark.parametrize(
    "ccd_code,endpoint_name,expected_leaving_atom",
    [
        ("NAG", "C1", "O1"),
        ("0E6", "C11", "CL"),
    ],
)
def test_infer_ccd_leaving_atoms_for_documented_components(
    biotite_ccd_wrapper,
    ccd_code,
    endpoint_name,
    expected_leaving_atom,
):
    query = Query.model_validate(
        {
            "query_name": f"documented_{ccd_code}",
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "ccd_codes": ccd_code,
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "S",
                    "smiles": "N",
                },
            ],
            "covalent_bonds": [[["L", 1, endpoint_name], ["S", 1, "N1"]]],
        }
    )

    effective_query = infer_ccd_leaving_atoms(query, biotite_ccd_wrapper)

    assert effective_query.leaving_atoms == [Atom("L", 1, expected_leaving_atom)]


def test_infer_ccd_leaving_atoms_for_documented_ambiguous_component(
    biotite_ccd_wrapper,
):
    query = Query.model_validate(
        {
            "query_name": "documented_0OD",
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "ccd_codes": "0OD",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "S",
                    "smiles": "N",
                },
            ],
            "covalent_bonds": [[["L", 1, "RH"], ["S", 1, "N1"]]],
        }
    )

    with pytest.raises(ValueError, match="multiple endpoint-local") as exc_info:
        infer_ccd_leaving_atoms(query, biotite_ccd_wrapper)
    for atom_name in ("CL1", "CL2", "CL3"):
        assert atom_name in str(exc_info.value)
