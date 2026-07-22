from types import SimpleNamespace

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
    _ligand_atom_index_map,
    _resolve_ligand_reference_molecule,
    featurize_ligand_stereochemistry_guidance,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    tokenize_atom_array,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.model.structure import diffusion_module as diffusion_module_lib
from openfold3.core.model.structure.diffusion_module import SampleDiffusion
from openfold3.core.model.structure.ligand_stereochemistry import (
    _buffered_rdkit_bounds,
    _finite_guidance_update,
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
    query: Query,
    structure,
    ligand_chain_id: str,
) -> torch.Tensor:
    """Return one ligand reference conformer on the structure's global atom axis."""
    ligand_mask = structure.atom_array.chain_id == ligand_chain_id
    global_indices = np.flatnonzero(ligand_mask)
    reference = _resolve_ligand_reference_molecule(
        query,
        structure.processed_reference_mols,
        ligand_chain_id,
    )
    assert reference is not None
    index_map = _ligand_atom_index_map(
        reference,
        structure.atom_array[ligand_mask],
        global_indices,
    )
    conformer = reference.mol.GetConformer()
    coords = torch.zeros((len(structure.atom_array), 3), dtype=torch.float64)
    for reference_index, global_index in index_map.items():
        position = conformer.GetAtomPosition(reference_index)
        coords[global_index] = torch.tensor(
            [position.x, position.y, position.z], dtype=coords.dtype
        )
    return coords[None, None]


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
            "rdkit_bounds_index": torch.tensor([[0], [1]]),
            "rdkit_lower_bounds": torch.tensor([1.0]),
            "rdkit_upper_bounds": torch.tensor([1.0]),
            "rdkit_bounds_bond_mask": torch.tensor([True]),
            "rdkit_bounds_angle_mask": torch.tensor([False]),
            "rdkit_bounds_pair_vdw_cutoff": torch.tensor([2.0]),
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
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> torch.Tensor:
    """Return the differentiable flat-bottom energy used by the test oracle."""
    energy = torch.zeros_like(value)
    if lower is not None:
        energy = energy + torch.relu(lower.expand_as(value) - value)
    if upper is not None:
        energy = energy + torch.relu(value - upper.expand_as(value))
    return energy


def _reference_ligand_stereochemistry_energy(
    coords: torch.Tensor,
    features: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Build an autograd oracle for the production analytic guidance gradient."""
    energies = []
    posebusters_features = {
        "rdkit_bounds_index",
        "rdkit_lower_bounds",
        "rdkit_upper_bounds",
        "rdkit_bounds_bond_mask",
        "rdkit_bounds_angle_mask",
        "rdkit_bounds_pair_vdw_cutoff",
    }
    if posebusters_features.issubset(features):
        index, lower, upper = _buffered_rdkit_bounds(features)
        if index.shape[-1] > 0:
            diff = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
            distance = torch.linalg.norm(diff, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
            energies.append(
                defaults.POSEBUSTERS_WEIGHT
                * _reference_flat_bottom(distance, lower, upper).sum()
            )

    if {"chiral_atom_index", "chiral_atom_orientations"}.issubset(features):
        index = features["chiral_atom_index"].long()
        if index.shape[-1] > 0:
            orientations = features["chiral_atom_orientations"].bool()
            lower = torch.zeros_like(orientations, dtype=coords.dtype)
            upper = torch.zeros_like(orientations, dtype=coords.dtype)
            lower[orientations] = defaults.CHIRAL_BUFFER
            upper[orientations] = float("inf")
            lower[~orientations] = float("-inf")
            upper[~orientations] = -defaults.CHIRAL_BUFFER
            energies.append(
                defaults.CHIRAL_ATOM_WEIGHT
                * _reference_flat_bottom(
                    _reference_dihedral(coords, index), lower, upper
                ).sum()
            )

    if {"stereo_bond_index", "stereo_bond_orientations"}.issubset(features):
        index = features["stereo_bond_index"].long()
        if index.shape[-1] > 0:
            orientations = features["stereo_bond_orientations"].bool()
            lower = torch.zeros_like(orientations, dtype=coords.dtype)
            upper = torch.zeros_like(orientations, dtype=coords.dtype)
            lower[orientations] = torch.pi - defaults.STEREO_BOND_BUFFER
            upper[orientations] = float("inf")
            lower[~orientations] = float("-inf")
            upper[~orientations] = defaults.STEREO_BOND_BUFFER
            energies.append(
                defaults.STEREO_BOND_WEIGHT
                * _reference_flat_bottom(
                    torch.abs(_reference_dihedral(coords, index)), lower, upper
                ).sum()
            )

    if "planar_bond_index" in features:
        index = features["planar_bond_index"].long()
        if index.shape[-1] > 0:
            double_bond_index = index.T
            first_improper = double_bond_index[:, [1, 2, 3, 0]]
            second_improper = double_bond_index[:, [4, 5, 0, 3]]
            improper_index = torch.cat([first_improper, second_improper], dim=0).T
            upper = torch.full(
                (improper_index.shape[-1],),
                defaults.PLANAR_BOND_BUFFER,
                dtype=coords.dtype,
                device=coords.device,
            )
            energies.append(
                defaults.PLANAR_BOND_WEIGHT
                * _reference_flat_bottom(
                    torch.abs(_reference_dihedral(coords, improper_index)),
                    None,
                    upper,
                ).sum()
            )

    if not energies:
        return None
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
    ("smiles", "min_bounds", "min_chiral", "min_stereo", "min_planar"),
    [
        ("C[C@H](O)C(=O)O", 1, 1, 0, 0),
        ("N[C@@H](C)C(=O)O", 1, 1, 0, 0),
        ("C[C@H](O)[C@H](O)C", 1, 2, 0, 0),
        ("N[C@](F)(Cl)Br", 1, 4, 0, 0),
        ("F/C(Cl)=C(Br)/I", 1, 0, 1, 1),
        ("c1ccccc1", 1, 0, 0, 0),
        ("C1CCCCCCCCCCC1", 1, 0, 0, 0),
    ],
)
def test_ligand_stereochemistry_features_for_representative_ligands(
    smiles: str,
    min_bounds: int,
    min_chiral: int,
    min_stereo: int,
    min_planar: int,
):
    features = _features_for_smiles(smiles)

    assert features["ligand_stereochemistry_guidance_enabled"].item()
    assert features["rdkit_bounds_index"].shape[0] == 2
    assert features["rdkit_bounds_index"].shape[1] >= min_bounds
    assert features["rdkit_bounds_pair_vdw_cutoff"].shape == (
        features["rdkit_bounds_index"].shape[1],
    )
    assert features["chiral_atom_index"].shape[0] == 4
    assert features["chiral_atom_index"].shape[1] >= min_chiral
    assert features["stereo_bond_index"].shape[0] == 4
    assert features["stereo_bond_index"].shape[1] >= min_stereo
    assert features["planar_bond_index"].shape[0] == 6
    assert features["planar_bond_index"].shape[1] >= min_planar


@pytest.mark.parametrize(
    ("ligand_definition", "index_name", "orientation_name", "is_dihedral_trans"),
    [
        (
            {"smiles": "C[C@H](O)C(=O)O"},
            "chiral_atom_index",
            "chiral_atom_orientations",
            False,
        ),
        (
            {"smiles": "F/C(Cl)=C(Br)/I"},
            "stereo_bond_index",
            "stereo_bond_orientations",
            True,
        ),
    ],
)
def test_emitted_stereo_targets_match_processed_reference_molecule(
    ligand_definition: dict,
    index_name: str,
    orientation_name: str,
    is_dihedral_trans: bool,
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
    coords = _reference_coords_in_atom_array_order(query, structure, "L")
    index = features[index_name]
    orientations = features[orientation_name]

    assert index.shape[1] > 0
    dihedrals = _reference_dihedral(coords, index)
    if is_dihedral_trans:
        observed_orientations = dihedrals.abs() > torch.pi / 2
    else:
        assert torch.all(dihedrals.abs() > 1e-4)
        observed_orientations = dihedrals > 0

    torch.testing.assert_close(observed_orientations.flatten(), orientations)


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

    bounds = [set(pair) for pair in features["rdkit_bounds_index"].T.tolist()]
    chiral = [set(group) for group in features["chiral_atom_index"].T.tolist()]
    stereo = [set(group) for group in features["stereo_bond_index"].T.tolist()]
    planar = [set(group) for group in features["planar_bond_index"].T.tolist()]

    assert any(pair <= ligand_l for pair in bounds)
    assert any(pair <= ligand_m for pair in bounds)
    assert all(pair <= ligand_l or pair <= ligand_m for pair in bounds)
    assert chiral and all(group <= ligand_l for group in chiral)
    assert stereo and all(group <= ligand_m for group in stereo)
    assert planar and all(group <= ligand_m for group in planar)


def test_ligand_stereochemistry_features_are_empty_when_disabled():
    features = _features_for_smiles("C[C@H](O)C(=O)O", guidance=False)

    assert not features["ligand_stereochemistry_guidance_enabled"].item()
    assert features["rdkit_bounds_index"].shape == (2, 0)
    assert features["rdkit_bounds_pair_vdw_cutoff"].shape == (0,)
    assert features["chiral_atom_index"].shape == (4, 0)
    assert features["stereo_bond_index"].shape == (4, 0)
    assert features["planar_bond_index"].shape == (6, 0)


@pytest.mark.parametrize(
    ("smiles", "feature_name"),
    [
        ("CC(O)C(=O)O", "chiral_atom_index"),
        ("FC(Cl)=C(Br)I", "stereo_bond_index"),
    ],
)
def test_unspecified_stereochemistry_does_not_create_guidance_targets(
    smiles: str,
    feature_name: str,
):
    features = _features_for_smiles(smiles)

    assert features[feature_name].shape[1] == 0


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

    input_energy = _reference_ligand_stereochemistry_energy(coords, features)
    mirrored_energy = _reference_ligand_stereochemistry_energy(
        mirrored_coords, features
    )

    assert input_energy is not None
    assert mirrored_energy is not None
    assert input_energy < mirrored_energy


def test_posebusters_bounds_penalize_out_of_plane_aromatic_geometry():
    features = _features_for_smiles("c1ccccc1")
    angles = torch.arange(6) * torch.pi / 3
    planar = 1.4 * torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    planar = planar[None, None]
    distorted = planar.clone()
    distorted[..., 0, 2] = 1.0

    planar_energy = _reference_ligand_stereochemistry_energy(planar, features)
    distorted_energy = _reference_ligand_stereochemistry_energy(distorted, features)

    assert planar_energy == 0
    assert distorted_energy is not None
    assert distorted_energy > planar_energy


def test_ligand_stereochemistry_distance_gradient_matches_autograd():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    features = _distance_features()

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, features)
    assert energy is not None
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, features)

    assert analytic_gradient is not None
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
    features = _empty_feature_tensors()
    features.update(
        {
            "chiral_atom_index": torch.tensor([[0], [1], [2], [3]]),
            "chiral_atom_orientations": torch.tensor([True]),
        }
    )

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, features)
    assert energy is not None
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, features)

    assert analytic_gradient is not None
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


def test_ligand_stereochemistry_guidance_accepts_collated_index_orientation():
    coords = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]
    )
    batch = _empty_guidance_batch()
    batch["chiral_atom_index"] = torch.tensor([[0, 1, 2, 3]])
    batch["chiral_atom_orientations"] = torch.tensor([True])
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


def test_ligand_stereochemistry_guidance_update_keeps_previous_nonfinite_particles():
    coords = torch.zeros((1, 1, 2, 3))
    previous = torch.ones_like(coords)
    gradient = torch.tensor([[[[float("inf"), 0.0, 0.0], [10.0, 0.0, 0.0]]]])

    guided = _finite_guidance_update(coords, gradient, previous)

    assert torch.isfinite(guided).all()
    torch.testing.assert_close(guided, previous)


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


def test_resolve_ligand_reference_molecule_uses_query_construction_order():
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
                    "molecule_type": "rna",
                    "chain_ids": ["R"],
                    "sequence": "AG",
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

    ligand_l = _resolve_ligand_reference_molecule(
        query, structure.processed_reference_mols, "L"
    )
    ligand_m = _resolve_ligand_reference_molecule(
        query, structure.processed_reference_mols, "M"
    )

    assert ligand_l is structure.processed_reference_mols[2]
    assert ligand_m is structure.processed_reference_mols[5]


def test_resolve_ligand_reference_molecule_handles_repeated_chain_ids():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["L", "M"],
                    "smiles": "C[C@H](O)C(=O)O",
                }
            ],
            "ligand_stereochemistry_guidance": True,
        }
    )
    structure = structure_with_ref_mols_from_query(query)

    first = _resolve_ligand_reference_molecule(
        query, structure.processed_reference_mols, "L"
    )
    second = _resolve_ligand_reference_molecule(
        query, structure.processed_reference_mols, "M"
    )

    assert first is structure.processed_reference_mols[0]
    assert second is structure.processed_reference_mols[1]


def test_resolve_ligand_reference_molecule_rejects_missing_polymer_sequence():
    query = SimpleNamespace(
        chains=[
            SimpleNamespace(
                molecule_type=MoleculeType.PROTEIN,
                chain_ids=["A"],
                sequence=None,
            )
        ]
    )

    with pytest.raises(ValueError, match="Chain A has no sequence"):
        _resolve_ligand_reference_molecule(query, [], "L")


def test_resolve_ligand_reference_molecule_rejects_short_reference_list():
    query = _ligand_query("CC")

    with pytest.raises(ValueError, match="Not enough processed reference molecules"):
        _resolve_ligand_reference_molecule(query, [], "L")


def test_resolve_ligand_reference_molecule_returns_none_for_unknown_chain():
    query = _ligand_query("CC")
    structure = structure_with_ref_mols_from_query(query)

    assert (
        _resolve_ligand_reference_molecule(
            query, structure.processed_reference_mols, "X"
        )
        is None
    )


def test_resolve_ligand_reference_molecule_rejects_unsupported_molecule_type():
    query = SimpleNamespace(
        chains=[
            SimpleNamespace(
                molecule_type=object(),
                chain_ids=["X"],
                sequence=None,
            )
        ]
    )

    with pytest.raises(ValueError, match="Unsupported molecule type"):
        _resolve_ligand_reference_molecule(query, [], "L")


def test_ligand_atom_index_map_uses_atom_names_instead_of_position():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array
    order = np.arange(len(atom_array))[::-1]

    idx_map = _ligand_atom_index_map(
        structure.processed_reference_mols[0],
        atom_array[order],
        order,
    )

    reference_names = [
        atom.GetProp("annot_atom_name")
        for atom in structure.processed_reference_mols[0].mol.GetAtoms()
    ]
    expected_by_name = dict(zip(atom_array.atom_name, np.arange(len(atom_array))))
    assert idx_map == {
        idx: int(expected_by_name[name]) for idx, name in enumerate(reference_names)
    }


def test_ligand_atom_index_map_supports_cropped_reference_atoms():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    reference = structure.processed_reference_mols[0]
    crop_mask = np.ones(reference.mol.GetNumAtoms(), dtype=bool)
    crop_mask[1] = False
    cropped_reference = ProcessedReferenceMolecule(
        mol=reference.mol,
        in_crop_mask=crop_mask,
    )
    ligand_atom_array = structure.atom_array[crop_mask]
    global_indices = np.flatnonzero(crop_mask)

    idx_map = _ligand_atom_index_map(
        cropped_reference, ligand_atom_array, global_indices
    )

    assert 1 not in idx_map
    assert set(idx_map.values()) == set(global_indices)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        ("mask", "reference mask"),
        ("crop_count", "atom names"),
        ("indices", "atom indices"),
        ("annotation", "annotated atom names"),
        ("names", "atom names"),
        ("elements", "atom elements"),
    ],
)
def test_ligand_atom_index_map_rejects_inconsistent_inputs(mutator: str, match: str):
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    reference = structure.processed_reference_mols[0]
    atom_array = structure.atom_array.copy()
    global_indices = np.arange(len(atom_array))

    if mutator == "mask":
        reference = ProcessedReferenceMolecule(
            mol=reference.mol,
            in_crop_mask=np.ones(reference.mol.GetNumAtoms() - 1, dtype=bool),
        )
    elif mutator == "crop_count":
        crop_mask = np.ones(reference.mol.GetNumAtoms(), dtype=bool)
        crop_mask[0] = False
        reference = ProcessedReferenceMolecule(
            mol=reference.mol,
            in_crop_mask=crop_mask,
        )
    elif mutator == "indices":
        global_indices = global_indices[:-1]
    elif mutator == "annotation":
        mol = Chem.Mol(reference.mol)
        mol.GetAtomWithIdx(0).ClearProp("annot_atom_name")
        reference = ProcessedReferenceMolecule(
            mol=mol,
            in_crop_mask=reference.in_crop_mask,
        )
    elif mutator == "names":
        atom_array.atom_name[0] = "BAD"
    elif mutator == "elements":
        atom_array.element[0] = "N" if atom_array.element[0] != "N" else "C"

    with pytest.raises(ValueError, match=match):
        _ligand_atom_index_map(reference, atom_array, global_indices)


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
    assert _compute_ligand_constraints(mol, {}) == featurization._empty_constraints()


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


def test_featurization_rejects_missing_ligand_atoms():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    structure.atom_array.chain_id[:] = "X"

    with pytest.raises(ValueError, match="could not find ligand chain"):
        featurize_ligand_stereochemistry_guidance(
            query, structure.atom_array, structure.processed_reference_mols
        )


def test_featurization_rejects_missing_reference_molecule(monkeypatch):
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)
    monkeypatch.setattr(
        featurization,
        "_resolve_ligand_reference_molecule",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match="processed reference molecule"):
        featurize_ligand_stereochemistry_guidance(
            query, structure.atom_array, structure.processed_reference_mols
        )


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
    assert (
        prepare_ligand_stereochemistry_guidance(
            _empty_guidance_batch(), torch.ones(1, 2)
        )
        is not None
    )


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
    del batch["rdkit_bounds_pair_vdw_cutoff"]

    with pytest.raises(ValueError, match="rdkit_bounds_pair_vdw_cutoff"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_non_vector_constraints():
    batch = _empty_guidance_batch()
    batch["rdkit_lower_bounds"] = torch.empty((2, 0))

    with pytest.raises(ValueError, match="1D constraint tensor"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_prepare_guidance_rejects_out_of_range_indices(bad_index):
    batch = _empty_guidance_batch()
    batch["rdkit_bounds_index"] = torch.tensor([[0], [bad_index]])
    for feature_name in (
        "rdkit_lower_bounds",
        "rdkit_upper_bounds",
        "rdkit_bounds_bond_mask",
        "rdkit_bounds_angle_mask",
        "rdkit_bounds_pair_vdw_cutoff",
    ):
        batch[feature_name] = torch.tensor([0])

    with pytest.raises(ValueError, match="out-of-range atom index"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_constraint_vector_length_mismatch():
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    batch["rdkit_upper_bounds"] = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="one value per rdkit_bounds_index"):
        prepare_ligand_stereochemistry_guidance(batch, torch.ones(1, 2))


def test_buffered_rdkit_bounds_match_boltz_pair_cutoff_rules():
    features = {
        "rdkit_bounds_index": torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
        "rdkit_lower_bounds": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "rdkit_upper_bounds": torch.tensor([4.0, 4.0, 4.0, 4.0]),
        "rdkit_bounds_bond_mask": torch.tensor([True, False, True, False]),
        "rdkit_bounds_angle_mask": torch.tensor([False, True, True, False]),
        "rdkit_bounds_pair_vdw_cutoff": torch.tensor([2.0, 2.0, 2.0, 2.0]),
    }

    index, lower, upper = _buffered_rdkit_bounds(features)

    torch.testing.assert_close(index, features["rdkit_bounds_index"])
    torch.testing.assert_close(lower, torch.tensor([0.875, 2.0, 0.875, 2.0]))
    torch.testing.assert_close(upper, torch.tensor([2.0, 4.5, 2.0, float("inf")]))


def test_flat_bottom_derivative_supports_upper_only_bound():
    values = torch.tensor([-2.0, 0.0, 2.0])

    upper_only = _flat_bottom_derivative(values, None, torch.tensor([1.0]))

    torch.testing.assert_close(upper_only, torch.tensor([0.0, 0.0, 1.0]))


def test_empty_constraint_features_have_no_energy_or_gradient():
    coords = torch.zeros(1, 1, 6, 3)
    features = _empty_feature_tensors()

    assert _reference_ligand_stereochemistry_energy(coords, features) is None
    assert ligand_stereochemistry_gradient(coords, features) is None


@pytest.mark.parametrize("is_e", [False, True])
def test_stereo_bond_gradient_matches_autograd(is_e):
    coords = _dihedral_coords()[..., :4, :]
    features = _empty_feature_tensors()
    features.update(
        {
            "stereo_bond_index": torch.tensor([[0], [1], [2], [3]]),
            "stereo_bond_orientations": torch.tensor([is_e]),
        }
    )
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, features)
    assert energy is not None
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, features)

    assert analytic_gradient is not None
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
    features = _empty_feature_tensors()
    features["planar_bond_index"] = torch.tensor([[0], [1], [2], [3], [4], [5]])
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_stereochemistry_energy(coords_with_grad, features)
    assert energy is not None
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_stereochemistry_gradient(coords, features)

    assert analytic_gradient is not None
    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-6, rtol=1e-6
    )


def test_apply_guidance_with_no_usable_constraints_is_a_noop():
    coords = torch.randn(1, 1, 2, 3, dtype=torch.float16)
    guidance = _prepare_guidance(_empty_guidance_batch(), num_atoms=2)

    guided = apply_ligand_stereochemistry_guidance(coords, guidance, step_fraction=1.0)

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
    del batch["chiral_atom_index"]

    with pytest.raises(ValueError, match="chiral_atom_index"):
        sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=1,
        )

    assert denoiser.calls == 0
