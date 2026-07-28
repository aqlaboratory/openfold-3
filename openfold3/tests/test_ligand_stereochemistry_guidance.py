from dataclasses import replace

import numpy as np
import pytest
import torch
from rdkit import Chem

from openfold3.core.config import ligand_stereochemistry_defaults as defaults
from openfold3.core.data.pipelines.featurization import (
    ligand_stereochemistry as featurization,
)
from openfold3.core.data.pipelines.featurization.ligand_stereochemistry import (
    _compute_chiral_atom_constraints,
    _compute_flatness_constraints,
    _compute_geometry_constraints,
    _compute_ligand_constraints,
    _compute_stereo_bond_constraints,
    _empty_feature_tensors,
    featurize_ligand_stereochemistry_guidance,
)
from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    tokenize_atom_array,
)
from openfold3.core.model.structure import diffusion_module as diffusion_module_lib
from openfold3.core.model.structure.diffusion_module import SampleDiffusion
from openfold3.core.model.structure.ligand_stereochemistry import (
    _flat_bottom_derivative,
    _normalize_index_tensor,
    _required_batch_scalar,
    apply_ligand_stereochemistry_guidance,
    ligand_stereochemistry_gradient,
    ligand_stereochemistry_guidance_enabled,
    prepare_ligand_stereochemistry_guidance,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


def _ligand_query(smiles: str, guidance: bool = True) -> Query:
    return Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["L"],
                    "smiles": smiles,
                }
            ],
            "ligand_stereochemistry_guidance": guidance,
        }
    )


def _features_for_smiles(smiles: str, guidance: bool = True) -> dict[str, torch.Tensor]:
    query = _ligand_query(smiles, guidance=guidance)
    structure = structure_with_ref_mols_from_query(query)
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    return featurize_ligand_stereochemistry_guidance(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
    )


def _reference_coords_in_atom_array_order(
    structure,
    ligand_chain_id: str,
) -> torch.Tensor:
    """Return one ligand reference conformer on the structure's global atom axis."""
    coords = torch.zeros((len(structure.atom_array), 3), dtype=torch.float64)
    global_indices = np.arange(len(structure.atom_array))
    for residue, reference in zip(
        residue_view_iter(structure.atom_array),
        structure.processed_reference_mols,
        strict=True,
    ):
        if residue.chain_id[0] != ligand_chain_id:
            continue
        conformer = reference.mol.GetConformer()
        reference_indices = np.flatnonzero(reference.in_crop_mask)
        residue_indices = global_indices[residue.indices]
        for reference_index, global_index in zip(
            reference_indices, residue_indices, strict=True
        ):
            position = conformer.GetAtomPosition(int(reference_index))
            coords[global_index] = torch.tensor(
                [position.x, position.y, position.z], dtype=coords.dtype
            )
        return coords[None, None]
    raise AssertionError(f"Ligand chain {ligand_chain_id!r} was not found")


def _empty_guidance_batch(
    *,
    enabled: bool = True,
    start_fraction: float = 0.0,
    num_gd_steps: int = 1,
) -> dict[str, torch.Tensor]:
    batch = _empty_feature_tensors()
    batch.update(
        {
            "ligand_stereochemistry_guidance_enabled": torch.tensor([enabled]),
            "ligand_stereochemistry_start_fraction": torch.tensor([start_fraction]),
            "ligand_stereochemistry_num_gd_steps": torch.tensor([num_gd_steps]),
        }
    )
    return batch


def _distance_features() -> dict[str, torch.Tensor]:
    features = _empty_feature_tensors()
    features.update(
        {
            "ligand_stereochemistry_distance_index": torch.tensor([[0], [1]]),
            "ligand_stereochemistry_distance_lower": torch.tensor([1.0]),
            "ligand_stereochemistry_distance_upper": torch.tensor([1.0]),
        }
    )
    return features


def _dihedral_coords() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.2, 0.1],
                    [0.1, 1.0, 0.3],
                    [0.2, 0.4, -1.0],
                    [1.2, 1.4, 0.8],
                    [-0.8, 0.7, 1.1],
                ]
            ]
        ],
        dtype=torch.float64,
    )


def _reference_dihedral(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Compute the test-oracle dihedral through differentiable PyTorch operations."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
    n_ijk = torch.cross(r_ij, r_kj, dim=-1)
    n_jkl = torch.cross(r_kj, r_kl, dim=-1)
    r_kj_norm = torch.linalg.norm(r_kj, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    sin_phi = (r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(dim=-1) / (
        r_kj_norm * n_ijk_norm * n_jkl_norm
    )
    cos_phi = (n_ijk * n_jkl).sum(dim=-1) / (n_ijk_norm * n_jkl_norm)
    return torch.atan2(sin_phi, cos_phi)


def _reference_flat_bottom(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Return the differentiable flat-bottom energy used by the test oracle."""
    return torch.relu(lower.expand_as(value) - value) + torch.relu(
        value - upper.expand_as(value)
    )


def _reference_ligand_stereochemistry_energy(
    coords: torch.Tensor,
    guidance,
) -> torch.Tensor:
    """Build an independent autograd oracle for the analytical gradient."""
    energies = []
    if guidance.distance.index.shape[1] > 0:
        index = guidance.distance.index
        diff = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        value = torch.linalg.norm(diff, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
        energies.append(
            (
                guidance.distance.weight
                * _reference_flat_bottom(
                    value, guidance.distance.lower, guidance.distance.upper
                )
            ).sum()
        )

    for restraints, absolute in (
        (guidance.signed_dihedral, False),
        (guidance.stereo_dihedral, True),
        (guidance.planar_dihedral, True),
    ):
        if restraints.index.shape[1] == 0:
            continue
        value = _reference_dihedral(coords, restraints.index)
        if absolute:
            value = value.abs()
        energies.append(
            (
                restraints.weight
                * _reference_flat_bottom(value, restraints.lower, restraints.upper)
            ).sum()
        )

    return sum(energies)


def _prepare_guidance(batch: dict[str, torch.Tensor], num_atoms: int):
    guidance = prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, num_atoms))
    assert guidance is not None
    return guidance


class _IdentityDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, xl_noisy, **_kwargs):
        self.calls += 1
        return xl_noisy


@pytest.mark.parametrize(
    (
        "smiles",
        "min_distances",
        "min_signed_dihedrals",
        "min_stereo_dihedrals",
        "min_planar_dihedrals",
    ),
    [
        ("C[C@H](O)C(=O)O", 1, 1, 0, 0),
        ("N[C@@H](C)C(=O)O", 1, 1, 0, 0),
        ("C[C@H](O)[C@H](O)C", 1, 2, 0, 0),
        ("N[C@](F)(Cl)Br", 1, 4, 0, 0),
        ("F/C(Cl)=C(Br)/I", 1, 0, 1, 2),
        ("c1ccccc1", 1, 0, 0, 0),
        ("C1CCCCCCCCCCC1", 1, 0, 0, 0),
    ],
)
def test_ligand_stereochemistry_features_for_representative_ligands(
    smiles: str,
    min_distances: int,
    min_signed_dihedrals: int,
    min_stereo_dihedrals: int,
    min_planar_dihedrals: int,
):
    features = _features_for_smiles(smiles)

    assert features["ligand_stereochemistry_guidance_enabled"].item()
    expected = {
        "distance": (2, min_distances),
        "signed_dihedral": (4, min_signed_dihedrals),
        "stereo_dihedral": (4, min_stereo_dihedrals),
        "planar_dihedral": (4, min_planar_dihedrals),
    }
    for name, (arity, minimum) in expected.items():
        prefix = f"ligand_stereochemistry_{name}"
        assert features[f"{prefix}_index"].shape[0] == arity
        assert features[f"{prefix}_index"].shape[1] >= minimum
        for suffix in ("lower", "upper"):
            assert features[f"{prefix}_{suffix}"].shape == (
                features[f"{prefix}_index"].shape[1],
            )


@pytest.mark.parametrize(
    ("ligand_definition", "restraint_name", "absolute"),
    [
        ({"smiles": "C[C@H](O)C(=O)O"}, "signed_dihedral", False),
        ({"smiles": "F/C(Cl)=C(Br)/I"}, "stereo_dihedral", True),
    ],
)
def test_emitted_stereo_targets_match_processed_reference_molecule(
    ligand_definition: dict,
    restraint_name: str,
    absolute: bool,
):
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["L"],
                    **ligand_definition,
                }
            ],
            "ligand_stereochemistry_guidance": True,
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    features = featurize_ligand_stereochemistry_guidance(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
    )
    coords = _reference_coords_in_atom_array_order(structure, "L")
    guidance = _prepare_guidance(features, num_atoms=len(structure.atom_array))
    restraints = getattr(guidance, restraint_name)

    assert restraints.index.shape[1] > 0
    dihedrals = _reference_dihedral(coords, restraints.index)
    if absolute:
        dihedrals = dihedrals.abs()
    assert torch.all(dihedrals >= restraints.lower)
    assert torch.all(dihedrals <= restraints.upper)


def test_multi_ligand_features_keep_constraints_on_their_source_ligand():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "AG",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["L"],
                    "smiles": "C[C@H](O)C(=O)O",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["M"],
                    "smiles": "F/C(Cl)=C(Br)/I",
                },
            ],
            "ligand_stereochemistry_guidance": True,
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    features = featurize_ligand_stereochemistry_guidance(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
    )
    ligand_l = set(np.flatnonzero(structure.atom_array.chain_id == "L"))
    ligand_m = set(np.flatnonzero(structure.atom_array.chain_id == "M"))

    bounds = [
        set(pair)
        for pair in features["ligand_stereochemistry_distance_index"].T.tolist()
    ]
    chiral = [
        set(group)
        for group in features["ligand_stereochemistry_signed_dihedral_index"].T.tolist()
    ]
    stereo = [
        set(group)
        for group in features["ligand_stereochemistry_stereo_dihedral_index"].T.tolist()
    ]
    planar = [
        set(group)
        for group in features["ligand_stereochemistry_planar_dihedral_index"].T.tolist()
    ]

    assert any(pair <= ligand_l for pair in bounds)
    assert any(pair <= ligand_m for pair in bounds)
    assert all(pair <= ligand_l or pair <= ligand_m for pair in bounds)
    assert chiral and all(group <= ligand_l for group in chiral)
    assert stereo and all(group <= ligand_m for group in stereo)
    assert planar and all(group <= ligand_m for group in planar)


def test_ligand_stereochemistry_features_are_empty_when_disabled():
    features = _features_for_smiles("C[C@H](O)C(=O)O", guidance=False)

    assert not features["ligand_stereochemistry_guidance_enabled"].item()
    assert features["ligand_stereochemistry_distance_index"].shape == (2, 0)
    assert features["ligand_stereochemistry_signed_dihedral_index"].shape == (4, 0)
    assert features["ligand_stereochemistry_stereo_dihedral_index"].shape == (4, 0)
    assert features["ligand_stereochemistry_planar_dihedral_index"].shape == (4, 0)


@pytest.mark.parametrize(
    ("smiles", "constraint_name"),
    [
        ("CC(O)C(=O)O", "chiral_index"),
        ("FC(Cl)=C(Br)I", "stereo_bond_index"),
    ],
)
def test_unspecified_stereochemistry_does_not_create_guidance_targets(
    smiles: str,
    constraint_name: str,
):
    mol = Chem.MolFromSmiles(smiles)
    constraints = _compute_ligand_constraints(
        mol, {atom.GetIdx(): atom.GetIdx() for atom in mol.GetAtoms()}
    )

    assert getattr(constraints, constraint_name) == []


def test_ligand_stereochemistry_energy_prefers_input_chirality():
    query = _ligand_query("C[C@H](O)C(=O)O")
    structure = structure_with_ref_mols_from_query(query)
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    features = featurize_ligand_stereochemistry_guidance(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
    )

    conformer = structure.processed_reference_mols[0].mol.GetConformer()
    coords = []
    for atom in structure.processed_reference_mols[0].mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    coords = torch.tensor(coords, dtype=torch.float32)[None, None]
    mirrored_coords = coords.clone()
    mirrored_coords[..., 0] *= -1
    guidance = _prepare_guidance(features, num_atoms=coords.shape[-2])

    input_energy = _reference_ligand_stereochemistry_energy(coords, guidance)
    mirrored_energy = _reference_ligand_stereochemistry_energy(
        mirrored_coords, guidance
    )

    assert input_energy < mirrored_energy


def test_posebusters_bounds_penalize_out_of_plane_aromatic_geometry():
    features = _features_for_smiles("c1ccccc1")
    guidance = _prepare_guidance(features, num_atoms=6)
    angles = torch.arange(6) * torch.pi / 3
    planar = 1.4 * torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    planar = planar[None, None]
    distorted = planar.clone()
    distorted[..., 0, 2] = 1.0

    planar_energy = _reference_ligand_stereochemistry_energy(planar, guidance)
    distorted_energy = _reference_ligand_stereochemistry_energy(distorted, guidance)

    assert planar_energy == 0
    assert distorted_energy > planar_energy


def test_ligand_stereochemistry_distance_gradient_matches_autograd():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, guidance)
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, guidance)

    torch.testing.assert_close(analytic_gradient, autograd_gradient)


def test_ligand_stereochemistry_dihedral_gradient_matches_autograd():
    coords = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.2, 0.1],
                    [0.1, 1.0, 0.3],
                    [0.2, 0.4, -1.0],
                ]
            ]
        ]
    )
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_stereochemistry_signed_dihedral_index": torch.tensor(
                [[0], [1], [2], [3]]
            ),
            "ligand_stereochemistry_signed_dihedral_lower": torch.tensor(
                [defaults.CHIRAL_BUFFER]
            ),
            "ligand_stereochemistry_signed_dihedral_upper": torch.tensor(
                [float("inf")]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, guidance)

    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-5, rtol=1e-5
    )


def test_ligand_stereochemistry_query_validates_guidance_settings():
    with pytest.raises(ValueError, match="start_fraction"):
        Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["L"],
                        "smiles": "CCO",
                    }
                ],
                "ligand_stereochemistry_start_fraction": 1.5,
            }
        )

    with pytest.raises(ValueError, match="num_gd_steps"):
        Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["L"],
                        "smiles": "CCO",
                    }
                ],
                "ligand_stereochemistry_num_gd_steps": 0,
            }
        )


def test_ligand_stereochemistry_guidance_is_noop_when_disabled():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(enabled=False)
    batch.update(_distance_features())
    guidance = prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=1.0)

    torch.testing.assert_close(guided, coords)


def test_ligand_stereochemistry_guidance_respects_start_fraction():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(start_fraction=0.9)
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=0.5)

    torch.testing.assert_close(guided, coords)


def test_ligand_stereochemistry_guidance_reduces_bond_violation():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(num_gd_steps=5)
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=0.5)

    initial_distance = torch.linalg.norm(coords[..., 0, :] - coords[..., 1, :])
    guided_distance = torch.linalg.norm(guided[..., 0, :] - guided[..., 1, :])
    assert guided_distance < initial_distance


def test_ligand_stereochemistry_guidance_accepts_collated_restraints():
    coords = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]
    )
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_stereochemistry_signed_dihedral_index": torch.tensor(
                [[0, 1, 2, 3]]
            ),
            "ligand_stereochemistry_signed_dihedral_lower": torch.tensor(
                [defaults.CHIRAL_BUFFER]
            ),
            "ligand_stereochemistry_signed_dihedral_upper": torch.tensor(
                [float("inf")]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=1.0)

    assert guided.shape == coords.shape
    assert torch.isfinite(guided).all()


def test_ligand_stereochemistry_guidance_runs_inside_inference_mode():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch()
    batch.update(_distance_features())

    with torch.inference_mode():
        coords = coords + 0.0
        batch = {
            key: value + 0 if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        guidance = _prepare_guidance(batch, num_atoms=2)
        guided = apply_ligand_stereochemistry_guidance(
            coords, guidance, step_fraction=1.0
        )

    assert guided.requires_grad is False
    assert not torch.equal(guided, coords)


def test_ligand_stereochemistry_query_uses_central_defaults():
    query = _ligand_query("CC", guidance=False)

    assert (
        query.ligand_stereochemistry_guidance
        == defaults.DEFAULT_LIGAND_STEREOCHEMISTRY_GUIDANCE_ENABLED
    )
    assert (
        query.ligand_stereochemistry_start_fraction
        == defaults.DEFAULT_LIGAND_STEREOCHEMISTRY_START_FRACTION
    )
    assert (
        query.ligand_stereochemistry_num_gd_steps
        == defaults.DEFAULT_LIGAND_STEREOCHEMISTRY_NUM_GD_STEPS
    )


def test_ligand_stereochemistry_query_requires_a_ligand_when_enabled():
    with pytest.raises(ValueError, match="requires at least one ligand"):
        Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": "AG",
                    }
                ],
                "ligand_stereochemistry_guidance": True,
            }
        )


def test_geometry_constraints_handle_single_atom_and_cropped_pairs():
    single_atom = Chem.MolFromSmiles("[Na+]")
    assert _compute_geometry_constraints(single_atom, {0: 4}) == (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    mol = Chem.MolFromSmiles("CCO")
    constraints = _compute_geometry_constraints(mol, {0: 10, 2: 12})
    assert constraints[0] == [(10, 12)]
    periodic_table = Chem.GetPeriodicTable()
    expected_vdw_cutoff = (
        defaults.VDW_PAIR_CUTOFF_OFFSET
        + (periodic_table.GetRvdw(6) + periodic_table.GetRvdw(8)) / 2
    )
    assert constraints[5] == pytest.approx([expected_vdw_cutoff])


def test_geometry_constraints_reject_unsupported_atomic_numbers():
    mol = Chem.MolFromSmiles("*C")

    with pytest.raises(ValueError, match="supported atomic numbers"):
        _compute_geometry_constraints(mol, {0: 0, 1: 1})


def test_constraint_extractors_handle_unassigned_and_uncropped_chemistry():
    mol = Chem.MolFromSmiles("C[C@H](O)C(=O)O")
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPRank"):
            atom.ClearProp("_CIPRank")
    assert _compute_chiral_atom_constraints(mol, {}) == ([], [])
    assert _compute_stereo_bond_constraints(mol, {}) == ([], [])
    assert (
        _compute_ligand_constraints(mol, {})
        == featurization._LigandGeometryConstraints()
    )


def test_constraint_extractors_skip_only_constraints_with_cropped_atoms():
    chiral_mol = Chem.MolFromSmiles("N[C@](F)(Cl)Br")
    Chem.AssignStereochemistry(chiral_mol, cleanIt=True, force=True)
    center_idx = Chem.FindMolChiralCenters(chiral_mol, includeUnassigned=False)[0][0]
    neighbors = sorted(
        chiral_mol.GetAtomWithIdx(center_idx).GetNeighbors(),
        key=lambda atom: int(atom.GetProp("_CIPRank")),
        reverse=True,
    )
    cropped_chiral_map = {
        atom.GetIdx(): atom.GetIdx()
        for atom in chiral_mol.GetAtoms()
        if atom.GetIdx() != neighbors[0].GetIdx()
    }

    chiral_constraints, _ = _compute_chiral_atom_constraints(
        chiral_mol, cropped_chiral_map
    )

    assert len(chiral_constraints) == 1

    stereo_mol = Chem.MolFromSmiles("F/C(Cl)=C(Br)/I")
    Chem.AssignStereochemistry(stereo_mol, cleanIt=True, force=True)
    stereo_bond = next(
        bond
        for bond in stereo_mol.GetBonds()
        if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}
    )
    start_idx = stereo_bond.GetBeginAtomIdx()
    end_idx = stereo_bond.GetEndAtomIdx()
    start_neighbors = sorted(
        (
            atom
            for atom in stereo_mol.GetAtomWithIdx(start_idx).GetNeighbors()
            if atom.GetIdx() != end_idx
        ),
        key=lambda atom: int(atom.GetProp("_CIPRank")),
        reverse=True,
    )
    full_map = {atom.GetIdx(): atom.GetIdx() for atom in stereo_mol.GetAtoms()}
    without_high_priority = full_map.copy()
    del without_high_priority[start_neighbors[0].GetIdx()]
    without_low_priority = full_map.copy()
    del without_low_priority[start_neighbors[1].GetIdx()]

    high_priority_cropped, _ = _compute_stereo_bond_constraints(
        stereo_mol, without_high_priority
    )
    low_priority_cropped, _ = _compute_stereo_bond_constraints(
        stereo_mol, without_low_priority
    )

    assert len(high_priority_cropped) == 1
    assert len(low_priority_cropped) == 1

    planar_match = stereo_mol.GetSubstructMatch(
        Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    )
    cropped_planar_map = full_map.copy()
    del cropped_planar_map[planar_match[0]]
    assert _compute_flatness_constraints(stereo_mol, cropped_planar_map) == []


def test_chiral_constraint_extractor_skips_non_tetrahedral_centers(monkeypatch):
    mol = Chem.MolFromSmiles("C=C")
    for rank, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("_CIPRank", rank)
    monkeypatch.setattr(
        Chem, "FindMolChiralCenters", lambda *_args, **_kwargs: [(0, "R")]
    )

    assert _compute_chiral_atom_constraints(mol, {0: 0, 1: 1}) == ([], [])


def test_compute_ligand_constraints_without_reference_conformer():
    mol = Chem.MolFromSmiles("CCO")

    constraints = _compute_ligand_constraints(mol, {0: 0, 1: 1, 2: 2})

    assert constraints.bounds_index
    assert constraints.chiral_index == []


def test_featurization_rejects_misaligned_reference_atoms():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    reference = structure.processed_reference_mols[0]
    malformed = replace(
        reference,
        in_crop_mask=np.arange(reference.mol.GetNumAtoms()) != 0,
    )

    with pytest.raises(ValueError, match="must match the OF3 residue"):
        featurize_ligand_stereochemistry_guidance(
            query, structure.atom_array, [malformed]
        )


def test_featurization_rejects_missing_reference_molecule():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        featurize_ligand_stereochemistry_guidance(query, structure.atom_array, [])


def test_normalize_index_tensor_handles_collated_and_transposed_shapes():
    collated = torch.tensor([[[[0], [1]]]])
    transposed = torch.tensor([[0, 1]])

    torch.testing.assert_close(
        _normalize_index_tensor(collated, 2, "index"),
        torch.tensor([[0], [1]]),
    )
    torch.testing.assert_close(
        _normalize_index_tensor(transposed, 2, "index"),
        torch.tensor([[0], [1]]),
    )


@pytest.mark.parametrize(
    ("tensor", "match"),
    [
        (torch.tensor([0, 1]), "2D index tensor"),
        (torch.tensor([[0, 1, 2]]), "index arity"),
        (torch.tensor([[0.0], [1.0]]), "integer atom indices"),
    ],
)
def test_normalize_index_tensor_rejects_malformed_indices(tensor, match):
    with pytest.raises(ValueError, match=match):
        _normalize_index_tensor(tensor, 2, "index")


def test_required_batch_scalar_accepts_python_values_and_rejects_vectors():
    assert _required_batch_scalar({"value": 3}, "value", int) == 3

    with pytest.raises(ValueError, match="must be scalar"):
        _required_batch_scalar({"value": torch.tensor([1.0, 2.0])}, "value", float)


def test_guidance_enabled_handles_absent_python_and_invalid_tensor_values():
    assert not ligand_stereochemistry_guidance_enabled({})
    assert ligand_stereochemistry_guidance_enabled(
        {"ligand_stereochemistry_guidance_enabled": True}
    )

    with pytest.raises(ValueError, match="must be scalar"):
        ligand_stereochemistry_guidance_enabled(
            {"ligand_stereochemistry_guidance_enabled": torch.tensor([True, False])}
        )


def test_prepare_guidance_accepts_complete_features_and_disabled_batches():
    assert prepare_ligand_stereochemistry_guidance({}, torch.ones(1, 2)) is None
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    assert prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2)) is not None


def test_prepare_guidance_rejects_multiple_queries():
    with pytest.raises(ValueError, match="one query per model batch"):
        prepare_ligand_stereochemistry_guidance(
            _empty_guidance_batch(), torch.ones(2, 2)
        )


@pytest.mark.parametrize(
    ("feature_name", "value", "match"),
    [
        (
            "ligand_stereochemistry_start_fraction",
            torch.tensor([-0.1]),
            "between 0 and 1",
        ),
        (
            "ligand_stereochemistry_start_fraction",
            torch.tensor([1.1]),
            "between 0 and 1",
        ),
        (
            "ligand_stereochemistry_num_gd_steps",
            torch.tensor([0]),
            "at least 1",
        ),
    ],
)
def test_prepare_guidance_rejects_invalid_settings(feature_name, value, match):
    batch = _empty_guidance_batch()
    batch[feature_name] = value

    with pytest.raises(ValueError, match=match):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_prepare_guidance_requires_every_emitted_feature():
    batch = _empty_guidance_batch()
    del batch["ligand_stereochemistry_distance_upper"]

    with pytest.raises(ValueError, match="ligand_stereochemistry_distance_upper"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_non_vector_constraints():
    batch = _empty_guidance_batch()
    batch["ligand_stereochemistry_distance_lower"] = torch.empty((2, 0))

    with pytest.raises(ValueError, match="1D constraint tensor"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_prepare_guidance_rejects_out_of_range_indices(bad_index):
    batch = _empty_guidance_batch()
    batch["ligand_stereochemistry_distance_index"] = torch.tensor([[0], [bad_index]])
    for feature_name in (
        "ligand_stereochemistry_distance_lower",
        "ligand_stereochemistry_distance_upper",
    ):
        batch[feature_name] = torch.tensor([0])

    with pytest.raises(ValueError, match="out-of-range atom index"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_constraint_vector_length_mismatch():
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    batch["ligand_stereochemistry_distance_upper"] = torch.tensor([1.0, 2.0])

    with pytest.raises(
        ValueError, match="one value per ligand_stereochemistry_distance_index"
    ):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_tensorized_distance_bounds_match_pair_cutoff_rules():
    constraints = featurization._LigandGeometryConstraints(
        bounds_index=[(0, 1), (0, 2), (0, 3), (0, 4)],
        lower_bounds=[1.0] * 4,
        upper_bounds=[4.0] * 4,
        bond_mask=[True, False, True, False],
        angle_mask=[False, True, True, False],
        pair_vdw_cutoffs=[2.0] * 4,
    )

    features = featurization._tensorize_constraints(constraints)

    torch.testing.assert_close(
        features["ligand_stereochemistry_distance_index"],
        torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
    )
    torch.testing.assert_close(
        features["ligand_stereochemistry_distance_lower"],
        torch.tensor([0.875, 2.0, 0.875, 2.0]),
    )
    torch.testing.assert_close(
        features["ligand_stereochemistry_distance_upper"],
        torch.tensor([2.0, 4.5, 2.0, float("inf")]),
    )


def test_flat_bottom_derivative_supports_upper_only_bound():
    values = torch.tensor([-2.0, 0.0, 2.0])

    upper_only = _flat_bottom_derivative(
        values, torch.tensor([float("-inf")]), torch.tensor([1.0])
    )

    torch.testing.assert_close(upper_only, torch.tensor([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("is_e", [False, True])
def test_stereo_bond_gradient_matches_autograd(is_e):
    coords = _dihedral_coords()[..., :4, :]
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_stereochemistry_stereo_dihedral_index": torch.tensor(
                [[0], [1], [2], [3]]
            ),
            "ligand_stereochemistry_stereo_dihedral_lower": torch.tensor(
                [torch.pi - defaults.STEREO_BOND_BUFFER if is_e else float("-inf")]
            ),
            "ligand_stereochemistry_stereo_dihedral_upper": torch.tensor(
                [float("inf") if is_e else defaults.STEREO_BOND_BUFFER]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, guidance)

    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-6, rtol=1e-6
    )


def test_dihedral_distinguishes_exact_cis_and_trans_geometry():
    index = torch.tensor([[0], [1], [2], [3]])
    cis = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]]
    )
    trans = cis.clone()
    trans[..., 3, 1] = -1.0

    torch.testing.assert_close(
        _reference_dihedral(cis, index).abs(), torch.zeros(1, 1, 1)
    )
    torch.testing.assert_close(
        _reference_dihedral(trans, index).abs(), torch.full((1, 1, 1), torch.pi)
    )


def test_planar_bond_gradient_matches_autograd():
    coords = _dihedral_coords()
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_stereochemistry_planar_dihedral_index": torch.tensor(
                [[1, 4], [2, 5], [3, 0], [0, 3]]
            ),
            "ligand_stereochemistry_planar_dihedral_lower": torch.full(
                (2,), float("-inf")
            ),
            "ligand_stereochemistry_planar_dihedral_upper": torch.full(
                (2,), defaults.PLANAR_BOND_BUFFER
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=6)
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, guidance)

    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-6, rtol=1e-6
    )


def test_apply_guidance_with_no_usable_constraints_is_a_noop():
    coords = torch.randn(1, 1, 2, 3, dtype=torch.float16)
    guidance = prepare_ligand_stereochemistry_guidance(
        _empty_guidance_batch(), torch.ones(1, 2)
    )

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=1.0)

    assert guidance is None
    torch.testing.assert_close(guided, coords)
    assert guided.dtype == coords.dtype


def test_sample_diffusion_reports_the_final_step_as_fraction_one(monkeypatch):
    observed_fractions = []

    def record_fraction(xl_denoised, guidance, step_fraction):
        assert guidance is not None
        observed_fractions.append(step_fraction)
        return xl_denoised

    monkeypatch.setattr(
        diffusion_module_lib,
        "apply_ligand_stereochemistry_guidance",
        record_fraction,
    )
    denoiser = _IdentityDenoiser()
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=denoiser,
    )
    batch = _empty_guidance_batch(start_fraction=1.0)
    batch.update(_distance_features())
    batch.update(
        {
            "atom_mask": torch.ones(1, 2),
            "token_mask": torch.ones(1, 1),
        }
    )

    with torch.no_grad():
        sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=1,
        )

    assert denoiser.calls == 2
    assert observed_fractions == [0.5, 1.0]


def test_sample_diffusion_disabled_guidance_matches_an_unmodified_batch():
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )
    common_batch = {
        "atom_mask": torch.ones(1, 2),
        "token_mask": torch.ones(1, 1),
    }
    disabled_batch = common_batch | _empty_guidance_batch(enabled=False)
    sampler_inputs = {
        "si_input": torch.zeros(1, 1, 1),
        "si_trunk": torch.zeros(1, 1, 1),
        "zij_trunk": torch.zeros(1, 1, 1, 1),
        "noise_schedule": torch.tensor([1.0, 0.5, 0.1]),
        "no_rollout_samples": 2,
    }

    with torch.no_grad():
        torch.manual_seed(7)
        unmodified = sampler(batch=common_batch, **sampler_inputs)
        torch.manual_seed(7)
        disabled = sampler(batch=disabled_batch, **sampler_inputs)

    torch.testing.assert_close(disabled, unmodified, rtol=0.0, atol=0.0)


def test_sample_diffusion_validates_guidance_before_denoising():
    denoiser = _IdentityDenoiser()
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=denoiser,
    )
    batch = _empty_guidance_batch()
    batch.update(
        {
            "atom_mask": torch.ones(1, 2),
            "token_mask": torch.ones(1, 1),
        }
    )
    del batch["ligand_stereochemistry_signed_dihedral_index"]

    with pytest.raises(ValueError, match="signed_dihedral_index"):
        sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=1,
        )

    assert denoiser.calls == 0
