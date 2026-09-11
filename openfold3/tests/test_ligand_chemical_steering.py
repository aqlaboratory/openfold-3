from dataclasses import replace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, BondList, BondType
from rdkit import Chem

from openfold3.core.config.ligand_chemical_steering_config import (
    LigandChemicalSteeringSettings,
)
from openfold3.core.data.pipelines.featurization import (
    ligand_chemical_steering as featurization,
)
from openfold3.core.data.pipelines.featurization.ligand_chemical_steering import (
    _compute_chiral_atom_constraints,
    _compute_flatness_constraints,
    _compute_geometry_constraints,
    _compute_ligand_constraints,
    _compute_stereo_bond_constraints,
    _compute_vdw_overlap_constraints,
    _empty_feature_tensors,
    featurize_ligand_chemical_steering,
)
from openfold3.core.data.primitives.structure.labels import residue_view_iter
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
from openfold3.core.model.structure.ligand_chemical_steering import (
    _flat_bottom_derivative,
    _normalize_index_tensor,
    _required_batch_scalar,
    apply_ligand_chemical_steering,
    ligand_chemical_steering_enabled,
    ligand_chemical_steering_gradient,
    prepare_ligand_chemical_steering,
)
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import InferenceExperimentConfig
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)

_DEFAULT_SETTINGS = LigandChemicalSteeringSettings()


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
            "ligand_chemical_steering": guidance,
        }
    )


def _features_for_smiles(
    smiles: str,
    guidance: bool = True,
    settings: LigandChemicalSteeringSettings = _DEFAULT_SETTINGS,
) -> dict[str, torch.Tensor]:
    query = _ligand_query(smiles, guidance=guidance)
    structure = structure_with_ref_mols_from_query(query)
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    return featurize_ligand_chemical_steering(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=settings,
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
    settings: LigandChemicalSteeringSettings = _DEFAULT_SETTINGS,
) -> dict[str, torch.Tensor]:
    batch = _empty_feature_tensors()
    batch.update(
        {
            "ligand_chemical_steering_enabled": torch.tensor([enabled]),
            "ligand_chemical_steering_start_fraction": torch.tensor([start_fraction]),
            "ligand_chemical_steering_num_gd_steps": torch.tensor([num_gd_steps]),
            "ligand_chemical_steering_vdw_guidance_interval": torch.tensor(
                [settings.vdw_guidance_interval]
            ),
            "ligand_chemical_steering_distance_weight": torch.tensor(
                [settings.distance_weight]
            ),
            "ligand_chemical_steering_vdw_overlap_weight": torch.tensor(
                [settings.vdw_weight]
            ),
            "ligand_chemical_steering_signed_dihedral_weight": torch.tensor(
                [settings.chiral_atom_weight]
            ),
            "ligand_chemical_steering_stereo_dihedral_weight": torch.tensor(
                [settings.stereo_bond_weight]
            ),
            "ligand_chemical_steering_planar_dihedral_weight": torch.tensor(
                [settings.planar_bond_weight]
            ),
        }
    )
    return batch


def _distance_features() -> dict[str, torch.Tensor]:
    features = _empty_feature_tensors()
    features.update(
        {
            "ligand_chemical_steering_distance_index": torch.tensor([[0], [1]]),
            "ligand_chemical_steering_distance_lower": torch.tensor([1.0]),
            "ligand_chemical_steering_distance_upper": torch.tensor([1.0]),
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
    r_kj_norm = torch.linalg.norm(r_kj, dim=-1).clamp_min(1e-6)
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1).clamp_min(1e-6)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1).clamp_min(1e-6)
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


def _reference_ligand_chemical_steering_energy(
    coords: torch.Tensor,
    guidance,
) -> torch.Tensor:
    """Build an independent autograd oracle for the analytical gradient."""
    energies = []
    if guidance.distance.atom_indices.shape[1] > 0:
        index = guidance.distance.atom_indices
        diff = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        value = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        energies.append(
            (
                guidance.distance.weight
                * _reference_flat_bottom(
                    value, guidance.distance.lower, guidance.distance.upper
                )
            ).sum()
        )

    if guidance.vdw_overlap.atom_indices.shape[1] > 0:
        index = guidance.vdw_overlap.atom_indices
        diff = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        value = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        energies.append(
            (
                guidance.vdw_overlap.weight
                * _reference_flat_bottom(
                    value,
                    guidance.vdw_overlap.lower,
                    guidance.vdw_overlap.upper,
                )
            ).sum()
        )

    for restraints, absolute in (
        (guidance.signed_dihedral, False),
        (guidance.stereo_dihedral, True),
        (guidance.planar_dihedral, True),
    ):
        if restraints.atom_indices.shape[1] == 0:
            continue
        value = _reference_dihedral(coords, restraints.atom_indices)
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
    guidance = prepare_ligand_chemical_steering(batch, torch.ones(1, num_atoms))
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
def test_ligand_chemical_steering_features_for_representative_ligands(
    smiles: str,
    min_distances: int,
    min_signed_dihedrals: int,
    min_stereo_dihedrals: int,
    min_planar_dihedrals: int,
):
    features = _features_for_smiles(smiles)

    assert features["ligand_chemical_steering_enabled"].item()
    expected = {
        "distance": (2, min_distances),
        "vdw_overlap": (2, 0),
        "signed_dihedral": (4, min_signed_dihedrals),
        "stereo_dihedral": (4, min_stereo_dihedrals),
        "planar_dihedral": (4, min_planar_dihedrals),
    }
    for name, (arity, minimum) in expected.items():
        prefix = f"ligand_chemical_steering_{name}"
        assert features[f"{prefix}_index"].shape[0] == arity
        assert features[f"{prefix}_index"].shape[1] >= minimum
        for suffix in ("lower", "upper"):
            assert features[f"{prefix}_{suffix}"].shape == (
                features[f"{prefix}_index"].shape[1],
            )


@pytest.mark.parametrize(
    ("smiles", "restraint_name", "absolute"),
    [
        ("C[C@H](O)C(=O)O", "signed_dihedral", False),
        ("F/C(Cl)=C(Br)/I", "stereo_dihedral", True),
    ],
)
def test_emitted_stereo_targets_match_processed_reference_molecule(
    smiles: str,
    restraint_name: str,
    absolute: bool,
):
    query = _ligand_query(smiles)
    structure = structure_with_ref_mols_from_query(query)
    features = featurize_ligand_chemical_steering(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
        _DEFAULT_SETTINGS,
    )
    coords = _reference_coords_in_atom_array_order(structure, "L")
    guidance = _prepare_guidance(features, num_atoms=len(structure.atom_array))
    restraints = getattr(guidance, restraint_name)

    assert restraints.atom_indices.shape[1] > 0
    dihedrals = _reference_dihedral(coords, restraints.atom_indices)
    if absolute:
        dihedrals = dihedrals.abs()
    assert torch.all(dihedrals >= restraints.lower)
    assert torch.all(dihedrals <= restraints.upper)


def test_pseudoasymmetric_centers_match_processed_reference_molecule():
    smiles = (
        "CS(=O)(=O)c1c(N[C@H]2CC[C@H](O)CC2)c(F)c(S(N)(=O)=O)c(F)c1"
        "N[C@H]1CC[C@@H](O)CC1"
    )
    query = _ligand_query(smiles)
    structure = structure_with_ref_mols_from_query(query)
    reference_mol = structure.processed_reference_mols[0].mol
    pseudoasymmetric_centers = Chem.FindMolChiralCenters(
        reference_mol,
        includeUnassigned=False,
        useLegacyImplementation=False,
    )
    features = featurize_ligand_chemical_steering(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
        _DEFAULT_SETTINGS,
    )
    coords = _reference_coords_in_atom_array_order(structure, "L")
    guidance = _prepare_guidance(features, num_atoms=len(structure.atom_array))
    restraints = guidance.signed_dihedral

    assert len(pseudoasymmetric_centers) == 4
    assert {orientation for _, orientation in pseudoasymmetric_centers} == {"r", "s"}
    assert restraints.atom_indices.shape == (4, 4)
    dihedrals = _reference_dihedral(coords, restraints.atom_indices)
    expected_positive = torch.isfinite(restraints.lower)
    assert torch.equal((dihedrals > 0).flatten(), expected_positive)


@pytest.mark.parametrize(
    "chain_order",
    [
        ("protein", "chiral_ligand", "stereo_ligand"),
        ("chiral_ligand", "protein", "stereo_ligand"),
        ("stereo_ligand", "chiral_ligand", "protein"),
    ],
    ids=("polymer-first", "ligand-first", "multiple-ligands-first"),
)
def test_multi_ligand_features_follow_reference_molecules_across_chain_order(
    chain_order: tuple[str, ...],
):
    chain_definitions = {
        "protein": {
            "molecule_type": "protein",
            "chain_ids": ["A"],
            "sequence": "AG",
        },
        "chiral_ligand": {
            "molecule_type": "ligand",
            "chain_ids": ["L"],
            "smiles": "C[C@H](O)C(=O)O",
        },
        "stereo_ligand": {
            "molecule_type": "ligand",
            "chain_ids": ["M"],
            "smiles": "F/C(Cl)=C(Br)/I",
        },
    }
    query = Query.model_validate(
        {
            "chains": [chain_definitions[name] for name in chain_order],
            "ligand_chemical_steering": True,
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    features = featurize_ligand_chemical_steering(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
        _DEFAULT_SETTINGS,
    )
    ligand_l = set(np.flatnonzero(structure.atom_array.chain_id == "L"))
    ligand_m = set(np.flatnonzero(structure.atom_array.chain_id == "M"))

    bounds = [
        set(pair)
        for pair in features["ligand_chemical_steering_distance_index"].T.tolist()
    ]
    chiral = [
        set(group)
        for group in features[
            "ligand_chemical_steering_signed_dihedral_index"
        ].T.tolist()
    ]
    stereo = [
        set(group)
        for group in features[
            "ligand_chemical_steering_stereo_dihedral_index"
        ].T.tolist()
    ]
    planar = [
        set(group)
        for group in features[
            "ligand_chemical_steering_planar_dihedral_index"
        ].T.tolist()
    ]
    vdw_overlap = features["ligand_chemical_steering_vdw_overlap_index"].T.tolist()

    assert any(pair <= ligand_l for pair in bounds)
    assert any(pair <= ligand_m for pair in bounds)
    assert all(pair <= ligand_l or pair <= ligand_m for pair in bounds)
    assert chiral and all(group <= ligand_l for group in chiral)
    assert stereo and all(group <= ligand_m for group in stereo)
    assert planar and all(group <= ligand_m for group in planar)
    assert vdw_overlap
    for atom_i, atom_j in vdw_overlap:
        assert (
            structure.atom_array.chain_id[atom_i]
            != structure.atom_array.chain_id[atom_j]
        )
        assert atom_i in ligand_l | ligand_m or atom_j in ligand_l | ligand_m


def test_ligand_chemical_steering_features_are_empty_when_disabled():
    features = _features_for_smiles("C[C@H](O)C(=O)O", guidance=False)

    assert not features["ligand_chemical_steering_enabled"].item()
    assert features["ligand_chemical_steering_distance_index"].shape == (2, 0)
    assert features["ligand_chemical_steering_vdw_overlap_index"].shape == (2, 0)
    assert features["ligand_chemical_steering_signed_dihedral_index"].shape == (4, 0)
    assert features["ligand_chemical_steering_stereo_dihedral_index"].shape == (4, 0)
    assert features["ligand_chemical_steering_planar_dihedral_index"].shape == (4, 0)


def test_zero_vdw_weight_skips_interchain_constraints():
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
                    "smiles": "CCO",
                },
            ],
            "ligand_chemical_steering": True,
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    settings = _DEFAULT_SETTINGS.model_copy(update={"vdw_weight": 0.0})

    features = featurize_ligand_chemical_steering(
        query,
        structure.atom_array,
        structure.processed_reference_mols,
        settings,
    )

    assert features["ligand_chemical_steering_vdw_overlap_index"].shape == (2, 0)


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
        mol,
        {atom.GetIdx(): atom.GetIdx() for atom in mol.GetAtoms()},
        _DEFAULT_SETTINGS,
    )

    assert getattr(constraints, constraint_name) == []


def test_ligand_chemical_steering_energy_prefers_input_chirality():
    query = _ligand_query("C[C@H](O)C(=O)O")
    structure = structure_with_ref_mols_from_query(query)
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    features = featurize_ligand_chemical_steering(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=_DEFAULT_SETTINGS,
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

    input_energy = _reference_ligand_chemical_steering_energy(coords, guidance)
    mirrored_energy = _reference_ligand_chemical_steering_energy(
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

    planar_energy = _reference_ligand_chemical_steering_energy(planar, guidance)
    distorted_energy = _reference_ligand_chemical_steering_energy(distorted, guidance)

    assert planar_energy == 0
    assert distorted_energy > planar_energy


def test_ligand_chemical_steering_distance_gradient_matches_autograd():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_chemical_steering_energy(coords_with_grad, guidance)
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_chemical_steering_gradient(coords, guidance)

    torch.testing.assert_close(analytic_gradient, autograd_gradient)


def test_vdw_overlap_gradient_matches_autograd_and_respects_boltz_interval():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_chemical_steering_vdw_overlap_index": torch.tensor([[0], [1]]),
            "ligand_chemical_steering_vdw_overlap_lower": torch.tensor([2.0]),
            "ligand_chemical_steering_vdw_overlap_upper": torch.tensor([float("inf")]),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=2)
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_chemical_steering_energy(coords_with_grad, guidance)
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    active_gradient = ligand_chemical_steering_gradient(
        coords, guidance, guidance_step=0
    )
    inactive_gradient = ligand_chemical_steering_gradient(
        coords, guidance, guidance_step=1
    )

    torch.testing.assert_close(active_gradient, autograd_gradient)
    torch.testing.assert_close(inactive_gradient, torch.zeros_like(coords))


def test_ligand_chemical_steering_dihedral_gradient_matches_autograd():
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
            "ligand_chemical_steering_signed_dihedral_index": torch.tensor(
                [[0], [1], [2], [3]]
            ),
            "ligand_chemical_steering_signed_dihedral_lower": torch.tensor(
                [_DEFAULT_SETTINGS.chiral_buffer]
            ),
            "ligand_chemical_steering_signed_dihedral_upper": torch.tensor(
                [float("inf")]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)

    coords_with_grad = coords.clone().requires_grad_(True)
    energy = _reference_ligand_chemical_steering_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_chemical_steering_gradient(coords, guidance)

    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-5, rtol=1e-5
    )


def test_ligand_chemical_steering_settings_validate_guidance_schedule():
    with pytest.raises(ValueError, match="start_fraction"):
        LigandChemicalSteeringSettings.model_validate({"start_fraction": 1.5})

    with pytest.raises(ValueError, match="num_gd_steps"):
        LigandChemicalSteeringSettings.model_validate({"num_gd_steps": 0})

    with pytest.raises(ValueError, match="vdw_guidance_interval"):
        LigandChemicalSteeringSettings.model_validate({"vdw_guidance_interval": 0})

    with pytest.raises(ValueError, match="extra_forbidden"):
        LigandChemicalSteeringSettings.model_validate({"unknown": 1})


def test_inference_experiment_records_and_routes_guidance_settings(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    experiment = InferenceExperimentConfig(
        inference_ckpt_path=checkpoint,
        cache_path=tmp_path,
        dataset_config_kwargs={
            "ligand_chemical_steering": {
                "start_fraction": 0.8,
                "num_gd_steps": 7,
                "chiral_atom_weight": 0.2,
            }
        },
    )
    serialized = experiment.model_dump()

    assert serialized["dataset_config_kwargs"]["ligand_chemical_steering"] == (
        experiment.dataset_config_kwargs.ligand_chemical_steering.model_dump()
    )

    runner = InferenceExperimentRunner(experiment)
    runner.inference_query_set = InferenceQuerySet(
        queries={"ligand": _ligand_query("C[C@H](O)C(=O)O")}
    )
    inference_job = runner.data_module_config.datasets[0].config

    assert (
        inference_job.ligand_chemical_steering
        == experiment.dataset_config_kwargs.ligand_chemical_steering
    )


def test_custom_guidance_settings_are_emitted_and_prepared():
    settings = LigandChemicalSteeringSettings(
        start_fraction=0.6,
        num_gd_steps=4,
        vdw_guidance_interval=3,
        distance_weight=0.02,
        vdw_weight=0.08,
        chiral_atom_weight=0.2,
        stereo_bond_weight=0.15,
        planar_bond_weight=0.12,
    )
    features = _features_for_smiles("C[C@H](O)C(=O)O", settings=settings)
    guidance = _prepare_guidance(features, num_atoms=6)

    assert guidance.start_fraction == pytest.approx(settings.start_fraction)
    assert guidance.num_gd_steps == settings.num_gd_steps
    assert guidance.vdw_guidance_interval == settings.vdw_guidance_interval
    assert guidance.distance.weight == pytest.approx(settings.distance_weight)
    assert guidance.vdw_overlap.weight == pytest.approx(settings.vdw_weight)
    assert guidance.signed_dihedral.weight == pytest.approx(settings.chiral_atom_weight)
    assert guidance.stereo_dihedral.weight == pytest.approx(settings.stereo_bond_weight)
    assert guidance.planar_dihedral.weight == pytest.approx(settings.planar_bond_weight)


def test_ligand_chemical_steering_is_noop_when_disabled():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(enabled=False)
    batch.update(_distance_features())
    guidance = prepare_ligand_chemical_steering(batch, torch.ones(1, 2))

    guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=1.0)

    torch.testing.assert_close(guided, coords)


def test_ligand_chemical_steering_respects_start_fraction():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(start_fraction=0.9)
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=0.5)

    torch.testing.assert_close(guided, coords)


def test_ligand_chemical_steering_reduces_bond_violation():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    batch = _empty_guidance_batch(num_gd_steps=5)
    batch.update(_distance_features())
    guidance = _prepare_guidance(batch, num_atoms=2)

    guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=0.5)

    initial_distance = torch.linalg.norm(coords[..., 0, :] - coords[..., 1, :])
    guided_distance = torch.linalg.norm(guided[..., 0, :] - guided[..., 1, :])
    assert guided_distance < initial_distance


def test_ligand_chemical_steering_accepts_collated_restraints():
    coords = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]
    )
    batch = _empty_guidance_batch()
    batch.update(
        {
            "ligand_chemical_steering_signed_dihedral_index": torch.tensor(
                [[0, 1, 2, 3]]
            ),
            "ligand_chemical_steering_signed_dihedral_lower": torch.tensor(
                [_DEFAULT_SETTINGS.chiral_buffer]
            ),
            "ligand_chemical_steering_signed_dihedral_upper": torch.tensor(
                [float("inf")]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)

    guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=1.0)

    assert guided.shape == coords.shape
    assert torch.isfinite(guided).all()


def test_ligand_chemical_steering_runs_inside_inference_mode():
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
        guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=1.0)

    assert guided.requires_grad is False
    assert not torch.equal(guided, coords)


def test_ligand_chemical_steering_query_only_controls_enablement():
    query = _ligand_query("CC", guidance=False)

    assert not query.ligand_chemical_steering
    assert "ligand_chemical_steering_start_fraction" not in Query.model_fields
    assert "ligand_chemical_steering_num_gd_steps" not in Query.model_fields


def test_ligand_chemical_steering_query_requires_a_ligand_when_enabled():
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
                "ligand_chemical_steering": True,
            }
        )


def test_geometry_constraints_handle_single_atom_and_cropped_pairs():
    single_atom = Chem.MolFromSmiles("[Na+]")
    assert (
        _compute_geometry_constraints(single_atom, {0: 4}, _DEFAULT_SETTINGS)
        == featurization._LigandGeometryConstraints()
    )

    mol = Chem.MolFromSmiles("CCO")
    constraints = _compute_geometry_constraints(mol, {0: 10, 2: 12}, _DEFAULT_SETTINGS)
    assert constraints.bounds_index == [(10, 12)]
    periodic_table = Chem.GetPeriodicTable()
    expected_vdw_cutoff = (
        _DEFAULT_SETTINGS.vdw_pair_cutoff_offset
        + (periodic_table.GetRvdw(6) + periodic_table.GetRvdw(8)) / 2
    )
    assert constraints.pair_vdw_cutoffs == pytest.approx([expected_vdw_cutoff])


def test_geometry_constraints_reject_unsupported_atomic_numbers():
    mol = Chem.MolFromSmiles("*C")

    with pytest.raises(ValueError, match="supported atomic numbers"):
        _compute_geometry_constraints(mol, {0: 0, 1: 1}, _DEFAULT_SETTINGS)


def test_vdw_overlap_constraints_follow_boltz_chain_pair_rules():
    atom_array = AtomArray(9)
    atom_array.chain_id = np.array(["A", "A", "B", "B", "L", "L", "M", "M", "I"])
    atom_array.element = np.array(["C", "N", "C", "O", "C", "O", "N", "C", "Na"])
    atom_array.set_annotation(
        "molecule_type_id",
        np.array(
            [
                int(MoleculeType.PROTEIN),
                int(MoleculeType.PROTEIN),
                int(MoleculeType.PROTEIN),
                int(MoleculeType.PROTEIN),
                int(MoleculeType.LIGAND),
                int(MoleculeType.LIGAND),
                int(MoleculeType.LIGAND),
                int(MoleculeType.LIGAND),
                int(MoleculeType.LIGAND),
            ]
        ),
    )
    atom_array.bonds = BondList(
        len(atom_array),
        np.array(
            [
                [0, 6, int(BondType.SINGLE)],
                [2, 4, int(BondType.COORDINATION)],
            ]
        ),
    )

    constraints = _compute_vdw_overlap_constraints(atom_array, _DEFAULT_SETTINGS)

    expected_chain_pairs = (
        ((0, 1), (4, 5)),
        ((2, 3), (4, 5)),
        ((2, 3), (6, 7)),
        ((4, 5), (6, 7)),
    )
    expected_pairs = {
        (atom_i, atom_j)
        for chain_i, chain_j in expected_chain_pairs
        for atom_i in chain_i
        for atom_j in chain_j
    }
    assert set(constraints.vdw_overlap_index) == expected_pairs
    assert all(8 not in pair for pair in constraints.vdw_overlap_index)
    for (atom_i, atom_j), lower_bound in zip(
        constraints.vdw_overlap_index,
        constraints.vdw_overlap_lower_bounds,
        strict=True,
    ):
        radii = [
            Chem.GetPeriodicTable().GetRvdw(
                Chem.GetPeriodicTable().GetAtomicNumber(
                    str(atom_array.element[atom_index]).capitalize()
                )
            )
            for atom_index in (atom_i, atom_j)
        ]
        assert lower_bound == pytest.approx(
            sum(radii) * (1.0 - _DEFAULT_SETTINGS.vdw_buffer)
        )


def test_vdw_overlap_constraints_reject_unsupported_elements():
    atom_array = AtomArray(4)
    atom_array.chain_id = np.array(["A", "A", "L", "L"])
    atom_array.element = np.array(["C", "N", "C", "*"])
    atom_array.set_annotation(
        "molecule_type_id",
        np.array(
            [
                int(MoleculeType.PROTEIN),
                int(MoleculeType.PROTEIN),
                int(MoleculeType.LIGAND),
                int(MoleculeType.LIGAND),
            ]
        ),
    )
    atom_array.bonds = BondList(len(atom_array))

    with pytest.raises(ValueError, match="supported elements"):
        _compute_vdw_overlap_constraints(atom_array, _DEFAULT_SETTINGS)


def test_constraint_extractors_handle_unassigned_and_uncropped_chemistry():
    mol = Chem.MolFromSmiles("C[C@H](O)C(=O)O")
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPRank"):
            atom.ClearProp("_CIPRank")
    assert (
        _compute_chiral_atom_constraints(mol, {})
        == featurization._LigandGeometryConstraints()
    )
    assert (
        _compute_stereo_bond_constraints(mol, {})
        == featurization._LigandGeometryConstraints()
    )
    assert (
        _compute_ligand_constraints(mol, {}, _DEFAULT_SETTINGS)
        == featurization._LigandGeometryConstraints()
    )


def test_constraint_extractors_skip_only_constraints_with_cropped_atoms():
    chiral_structure = structure_with_ref_mols_from_query(
        _ligand_query("N[C@](F)(Cl)Br")
    )
    chiral_mol = chiral_structure.processed_reference_mols[0].mol
    center = next(
        atom
        for atom in chiral_mol.GetAtoms()
        if atom.GetChiralTag()
        in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
    )
    neighbors = sorted(center.GetNeighbors(), key=lambda atom: atom.GetIdx())
    cropped_chiral_map = {
        atom.GetIdx(): atom.GetIdx()
        for atom in chiral_mol.GetAtoms()
        if atom.GetIdx() != neighbors[0].GetIdx()
    }

    chiral_constraints = _compute_chiral_atom_constraints(
        chiral_mol, cropped_chiral_map
    )

    assert len(chiral_constraints.chiral_index) == 1

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

    high_priority_cropped = _compute_stereo_bond_constraints(
        stereo_mol, without_high_priority
    )
    low_priority_cropped = _compute_stereo_bond_constraints(
        stereo_mol, without_low_priority
    )

    assert len(high_priority_cropped.stereo_bond_index) == 1
    assert len(low_priority_cropped.stereo_bond_index) == 1

    planar_match = stereo_mol.GetSubstructMatch(
        Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    )
    cropped_planar_map = full_map.copy()
    del cropped_planar_map[planar_match[0]]
    assert (
        _compute_flatness_constraints(stereo_mol, cropped_planar_map)
        == featurization._LigandGeometryConstraints()
    )


def test_chiral_constraint_extractor_skips_non_tetrahedral_centers():
    mol = Chem.MolFromSmiles("C=C")
    mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    assert (
        _compute_chiral_atom_constraints(mol, {0: 0, 1: 1})
        == featurization._LigandGeometryConstraints()
    )


def test_chiral_constraint_extractor_requires_reference_conformer():
    mol = Chem.MolFromSmiles("N[C@](F)(Cl)Br")
    idx_map = {atom.GetIdx(): atom.GetIdx() for atom in mol.GetAtoms()}

    with pytest.raises(ValueError, match="require an OF3 reference conformer"):
        _compute_chiral_atom_constraints(mol, idx_map)


def test_compute_ligand_constraints_without_reference_conformer():
    mol = Chem.MolFromSmiles("CCO")

    constraints = _compute_ligand_constraints(
        mol, {0: 0, 1: 1, 2: 2}, _DEFAULT_SETTINGS
    )

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
        featurize_ligand_chemical_steering(
            query, structure.atom_array, [malformed], _DEFAULT_SETTINGS
        )


def test_featurization_rejects_missing_reference_molecule():
    query = _ligand_query("CCO")
    structure = structure_with_ref_mols_from_query(query)

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        featurize_ligand_chemical_steering(
            query, structure.atom_array, [], _DEFAULT_SETTINGS
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
    assert not ligand_chemical_steering_enabled({})
    assert ligand_chemical_steering_enabled({"ligand_chemical_steering_enabled": True})

    with pytest.raises(ValueError, match="must be scalar"):
        ligand_chemical_steering_enabled(
            {"ligand_chemical_steering_enabled": torch.tensor([True, False])}
        )


def test_prepare_guidance_accepts_complete_features_and_disabled_batches():
    assert prepare_ligand_chemical_steering({}, torch.ones(1, 2)) is None
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    assert prepare_ligand_chemical_steering(batch, torch.ones(1, 2)) is not None


def test_prepare_guidance_rejects_multiple_queries():
    with pytest.raises(ValueError, match="one query per model batch"):
        prepare_ligand_chemical_steering(_empty_guidance_batch(), torch.ones(2, 2))


@pytest.mark.parametrize(
    ("feature_name", "value", "match"),
    [
        (
            "ligand_chemical_steering_start_fraction",
            torch.tensor([-0.1]),
            "between 0 and 1",
        ),
        (
            "ligand_chemical_steering_start_fraction",
            torch.tensor([1.1]),
            "between 0 and 1",
        ),
        (
            "ligand_chemical_steering_num_gd_steps",
            torch.tensor([0]),
            "at least 1",
        ),
        (
            "ligand_chemical_steering_vdw_guidance_interval",
            torch.tensor([0]),
            "at least 1",
        ),
        (
            "ligand_chemical_steering_distance_weight",
            torch.tensor([-0.1]),
            "finite non-negative",
        ),
    ],
)
def test_prepare_guidance_rejects_invalid_settings(feature_name, value, match):
    batch = _empty_guidance_batch()
    batch[feature_name] = value

    with pytest.raises(ValueError, match=match):
        prepare_ligand_chemical_steering(batch, torch.ones(1, 2))


def test_prepare_guidance_requires_every_emitted_feature():
    batch = _empty_guidance_batch()
    del batch["ligand_chemical_steering_distance_upper"]

    with pytest.raises(ValueError, match="ligand_chemical_steering_distance_upper"):
        prepare_ligand_chemical_steering(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_non_vector_constraints():
    batch = _empty_guidance_batch()
    batch["ligand_chemical_steering_distance_lower"] = torch.empty((2, 0))

    with pytest.raises(ValueError, match="1D constraint tensor"):
        prepare_ligand_chemical_steering(batch, torch.ones(1, 2))


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_prepare_guidance_rejects_out_of_range_indices(bad_index):
    batch = _empty_guidance_batch()
    batch["ligand_chemical_steering_distance_index"] = torch.tensor([[0], [bad_index]])
    for feature_name in (
        "ligand_chemical_steering_distance_lower",
        "ligand_chemical_steering_distance_upper",
    ):
        batch[feature_name] = torch.tensor([0])

    with pytest.raises(ValueError, match="out-of-range atom index"):
        prepare_ligand_chemical_steering(batch, torch.ones(1, 2))


def test_prepare_guidance_rejects_constraint_vector_length_mismatch():
    batch = _empty_guidance_batch()
    batch.update(_distance_features())
    batch["ligand_chemical_steering_distance_upper"] = torch.tensor([1.0, 2.0])

    with pytest.raises(
        ValueError, match="one value per ligand_chemical_steering_distance_index"
    ):
        prepare_ligand_chemical_steering(batch, torch.ones(1, 2))


def test_tensorized_distance_bounds_match_pair_cutoff_rules():
    constraints = featurization._LigandGeometryConstraints(
        bounds_index=[(0, 1), (0, 2), (0, 3), (0, 4)],
        lower_bounds=[1.0] * 4,
        upper_bounds=[4.0] * 4,
        bond_mask=[True, False, True, False],
        angle_mask=[False, True, True, False],
        pair_vdw_cutoffs=[2.0] * 4,
    )

    features = featurization._tensorize_constraints(constraints, _DEFAULT_SETTINGS)

    torch.testing.assert_close(
        features["ligand_chemical_steering_distance_index"],
        torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
    )
    torch.testing.assert_close(
        features["ligand_chemical_steering_distance_lower"],
        torch.tensor([0.875, 2.0, 0.875, 2.0]),
    )
    torch.testing.assert_close(
        features["ligand_chemical_steering_distance_upper"],
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
            "ligand_chemical_steering_stereo_dihedral_index": torch.tensor(
                [[0], [1], [2], [3]]
            ),
            "ligand_chemical_steering_stereo_dihedral_lower": torch.tensor(
                [
                    torch.pi - _DEFAULT_SETTINGS.stereo_bond_buffer
                    if is_e
                    else float("-inf")
                ]
            ),
            "ligand_chemical_steering_stereo_dihedral_upper": torch.tensor(
                [float("inf") if is_e else _DEFAULT_SETTINGS.stereo_bond_buffer]
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=4)
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_chemical_steering_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_chemical_steering_gradient(coords, guidance)

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
            "ligand_chemical_steering_planar_dihedral_index": torch.tensor(
                [[1, 4], [2, 5], [3, 0], [0, 3]]
            ),
            "ligand_chemical_steering_planar_dihedral_lower": torch.full(
                (2,), float("-inf")
            ),
            "ligand_chemical_steering_planar_dihedral_upper": torch.full(
                (2,), _DEFAULT_SETTINGS.planar_bond_buffer
            ),
        }
    )
    guidance = _prepare_guidance(batch, num_atoms=6)
    coords_with_grad = coords.clone().requires_grad_(True)

    energy = _reference_ligand_chemical_steering_energy(coords_with_grad, guidance)
    assert energy > 0
    (autograd_gradient,) = torch.autograd.grad(energy, coords_with_grad)
    analytic_gradient = ligand_chemical_steering_gradient(coords, guidance)

    torch.testing.assert_close(
        analytic_gradient, autograd_gradient, atol=1e-6, rtol=1e-6
    )


def test_apply_guidance_with_no_usable_constraints_is_a_noop():
    coords = torch.randn(1, 1, 2, 3, dtype=torch.float16)
    guidance = prepare_ligand_chemical_steering(
        _empty_guidance_batch(), torch.ones(1, 2)
    )

    guided = apply_ligand_chemical_steering(coords, guidance, step_fraction=1.0)

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
        "apply_ligand_chemical_steering",
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
    del batch["ligand_chemical_steering_signed_dihedral_index"]

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
