import torch

from openfold3.core.data.framework.single_datasets.inference import (
    InferenceDataset,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import get_token_count
from openfold3.core.model.structure.ligand_planarity import (
    _absolute_dihedrals_and_gradients,
    _planarity_violations,
    apply_ligand_planarity_restraints,
    prepare_ligand_planarity_restraints,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

ISSUE_136_SMILES = "O=C1NC(=S)N(c2cccc(Br)c2)C(=O)/C1=C/c1ccco1"


def _restraint_loss(coords, restraints):
    angles, _ = _absolute_dihedrals_and_gradients(coords, restraints.atom_indices)
    violations = _planarity_violations(angles, restraints.trans_orientations)
    return 0.5 * violations.square().sum()


def test_issue_136_implicit_hydrogen_alkene_is_fully_restrained():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "A",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["C"],
                    "smiles": ISSUE_136_SMILES,
                },
            ]
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    dataset = InferenceDataset.__new__(InferenceDataset)
    features = dataset.create_structure_features(
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        n_tokens=get_token_count(structure.atom_array),
    )
    ligand_start = int((structure.atom_array.chain_id != "C").sum())

    assert features["ligand_planarity_index"].T.tolist() == [
        [ligand_start + index for index in (13, 15, 16, 17)],
        [ligand_start + index for index in (1, 15, 16, 17)],
    ]
    assert features["ligand_planarity_trans"].tolist() == [True, False]

    processed_mol = structure.processed_reference_mols[-1]
    ligand_coords = torch.as_tensor(
        processed_mol.mol.GetConformer().GetPositions()[processed_mol.in_crop_mask],
        dtype=torch.float32,
    )
    coords = torch.zeros((len(structure.atom_array), 3))
    coords[ligand_start:] = ligand_coords
    central = ligand_start + 16
    normal = torch.cross(
        coords[ligand_start + 15] - coords[central],
        coords[ligand_start + 17] - coords[central],
        dim=0,
    )
    coords[ligand_start + 1] += normal / torch.linalg.norm(normal)
    restraints = prepare_ligand_planarity_restraints(
        features,
        atom_mask=torch.ones((1, len(coords))),
    )
    before = _restraint_loss(coords[None, None], restraints)
    guided = apply_ligand_planarity_restraints(coords[None, None], restraints)
    after = _restraint_loss(guided, restraints)

    assert before > 0
    assert after < 1e-6
    assert torch.equal(guided[0, 0, :ligand_start], coords[:ligand_start])
