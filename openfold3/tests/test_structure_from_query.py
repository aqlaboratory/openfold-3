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
import numpy as np
import pytest
from biotite.structure.io import pdbx
from rdkit import Chem

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.query import (
    StructureWithReferenceMolecules,
    processed_reference_molecule_from_mol,
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)

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
