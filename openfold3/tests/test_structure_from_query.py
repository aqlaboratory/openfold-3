# Copyright 2025 AlQuraishi Laboratory
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
import pickle
from pathlib import Path
from pydantic import ValidationError

import pytest
from rdkit import Chem

from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.primitives.structure.query import (
    processed_reference_molecule_from_mol,
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
    Chain,
    MoleculeType,
)
from openfold3.tests.custom_assert_utils import (
    assert_atomarray_equal,
    assert_ref_mols_equal,
)

reference_data_path = Path(__file__).parent / "test_data" / "structure_from_query"

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


@pytest.mark.parametrize(
    "query, ground_truth_file",
    [
        (
            standard_peptide_query,
            reference_data_path / "structure-w-ref-mols_std-peptide.pkl",
        ),
        (
            non_canonical_peptide_query,
            reference_data_path / "structure-w-ref-mols_non-std-peptide.pkl",
        ),
    ],
    ids=[
        "standard_peptide",
        "non_canonical_peptide",
    ],
)
def test_structure_with_ref_mols_from_query(query, ground_truth_file):
    """Tests the structure_with_ref_mols_from_query function."""
    structure_with_ref_mols = structure_with_ref_mols_from_query(query)

    # Get reference file
    structure_with_ref_mols_gt = pickle.loads(ground_truth_file.read_bytes())

    # Check that atom arrays match (for some reason the GT generation script generated a
    # different order of annotations but that's fine)
    assert_atomarray_equal(
        structure_with_ref_mols.atom_array,
        structure_with_ref_mols_gt.atom_array,
        strict_annot_order=False,
    )

    # Check that reference molecules match
    for ref_mol, ref_mol_gt in zip(
        structure_with_ref_mols.processed_reference_mols,
        structure_with_ref_mols_gt.processed_reference_mols,
        strict=False,
    ):
        assert_ref_mols_equal(ref_mol, ref_mol_gt)


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


def test_chain_residue_id_generation():
    """Tests the custom residue ID generation logic in the Chain model, verifying
    priority and validation of numbering schemes (residue_ids vs. starting_residue_number).
    """
    
    base_params = {
        "molecule_type": MoleculeType.PROTEIN,
        "chain_ids": ["A"],
        "sequence": "AAA" 
    }
    
    chain_default = Chain.model_validate(base_params)
    assert chain_default.residue_ids == ['1', '2', '3']
    
    params_start = base_params.copy()
    params_start['starting_residue_number'] = "100"
    
    chain_start = Chain.model_validate(params_start)
    assert chain_start.residue_ids == ['100', '101', '102']
    
    explicit_ids = ['1A', '2', '3B']
    params_explicit_priority = base_params.copy()
    params_explicit_priority['residue_ids'] = explicit_ids
    params_explicit_priority['starting_residue_number'] = "500" 
    
    chain_explicit = Chain.model_validate(params_explicit_priority)
    assert chain_explicit.residue_ids == explicit_ids
    
    params_mismatch = base_params.copy()
    params_mismatch['residue_ids'] = ['10', '11'] 
    
    with pytest.raises(ValueError, match="Length of residue_ids"): 
        Chain.model_validate(params_mismatch)
        
    params_invalid_start = base_params.copy()
    params_invalid_start['starting_residue_number'] = "invalid_number"
    
    with pytest.raises(ValueError, match="must be convertible to an integer"):
        Chain.model_validate(params_invalid_start)
