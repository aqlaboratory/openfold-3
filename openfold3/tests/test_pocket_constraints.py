import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from pydantic import ValidationError

from openfold3.core.config import pocket_sampling_config
from openfold3.core.config.pocket_sampling_config import PocketSamplingSettings
from openfold3.core.data.pipelines.featurization import pocket_constraints
from openfold3.core.data.pipelines.featurization.pocket_constraints import (
    create_pocket_sampling_features,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.model.structure import diffusion_module
from openfold3.core.model.structure.diffusion_module import (
    SampleDiffusion,
    _build_pocket_sampling_seeds,
    _feature_mask,
    create_noise_schedule,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


def _query_with_pocket_constraint() -> Query:
    return Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "AC",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "smiles": "CCO",
                },
            ],
            "pocket_constraint": {
                "ligand_chain_id": "L",
                "pocket_residues": [["A", 2]],
                "max_distance": 3.5,
            },
        }
    )


def _atom_array() -> AtomArray:
    atom_array: AtomArray = AtomArray(7)
    atom_array.coord = np.zeros((7, 3), dtype=float)
    atom_array.chain_id = np.array(["A", "A", "A", "A", "L", "L", "L"])
    atom_array.res_id = np.array([1, 1, 2, 2, 1, 1, 1])
    atom_array.res_name = np.array(["ALA", "ALA", "CYS", "CYS", "LIG", "LIG", "LIG"])
    atom_array.atom_name = np.array(["CA", "CB", "CA", "CB", "C1", "C2", "O1"])
    atom_array.element = np.array(["C", "N", "C", "O", "C", "C", "O"])
    atom_array.set_annotation(
        "molecule_type_id",
        np.array([int(MoleculeType.PROTEIN)] * 4 + [int(MoleculeType.LIGAND)] * 3),
    )
    return atom_array


def test_pocket_constraints_parse_without_unused_strength():
    query = _query_with_pocket_constraint()

    constraint = query.pocket_constraint
    assert constraint.ligand_chain_id == "L"
    assert constraint.pocket_residues[0].chain_id == "A"
    assert constraint.pocket_residues[0].residue_id == 2
    assert constraint.max_distance == 3.5


def test_pocket_constraints_default_max_distance():
    payload = _query_with_pocket_constraint().model_dump()
    del payload["pocket_constraint"]["max_distance"]

    query = Query.model_validate(payload)

    assert query.pocket_constraint.max_distance == pytest.approx(
        pocket_sampling_config.DEFAULT_POCKET_CONSTRAINT_MAX_DISTANCE
    )


def test_pocket_constraints_reject_unused_strength_field():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraint"]["strength"] = 1.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Query.model_validate(payload)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        (
            {"pocket_residues": []},
            "pocket_residues must contain at least one residue",
        ),
        ({"max_distance": 0.0}, "max_distance must be positive"),
    ],
)
def test_pocket_constraints_validate_constraint_fields(update, match):
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraint"].update(update)

    with pytest.raises(ValidationError, match=match):
        Query.model_validate(payload)


def test_pocket_constraints_ligand_chain_must_reference_ligand():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraint"]["ligand_chain_id"] = "A"

    with pytest.raises(ValidationError, match="does not match any ligand chain"):
        Query.model_validate(payload)


def test_create_pocket_sampling_features_uses_defaults():
    settings = PocketSamplingSettings(rdkit_num_conformers=0)

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=_atom_array(),
        settings=settings,
    )

    assert features["pocket_sampling_enabled"].item()
    assert features["pocket_sampling_ligand_atom_mask"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert features["pocket_sampling_pocket_atom_mask"].tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert features["pocket_sampling_contact_distance"].item() == pytest.approx(3.5)
    assert features["pocket_sampling_num_parents"].item() == settings.num_parents
    assert features["pocket_sampling_candidates"].item() == settings.candidates
    assert features["pocket_sampling_start_frac"].item() == pytest.approx(
        settings.noise_frac
    )
    assert features["pocket_sampling_ligand_jitter"].item() == pytest.approx(
        settings.ligand_jitter
    )
    assert features["pocket_sampling_center_jitter"].item() == pytest.approx(
        settings.center_jitter
    )
    assert features["pocket_sampling_surface_jitter"].item() == pytest.approx(
        settings.surface_jitter
    )
    assert features["pocket_sampling_vdw_buffer"].item() == pytest.approx(
        settings.vdw_buffer
    )
    assert features["pocket_sampling_diversity_rmsd"].item() == pytest.approx(
        settings.diversity_rmsd
    )
    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_without_constraint_is_noop():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "AC",
                },
            ],
        }
    )

    assert create_pocket_sampling_features(query=query, atom_array=_atom_array()) == {}


def test_create_pocket_sampling_features_rejects_missing_pocket_residue():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraint"]["pocket_residues"].append(("A", 99))

    with pytest.raises(ValueError, match="A:99"):
        create_pocket_sampling_features(
            query=Query.model_validate(payload),
            atom_array=_atom_array(),
            settings=PocketSamplingSettings(rdkit_num_conformers=0),
        )


def test_create_pocket_sampling_features_rejects_empty_ligand_mask():
    atom_array = _atom_array()
    atom_array.molecule_type_id[atom_array.chain_id == "L"] = MoleculeType.PROTEIN

    with pytest.raises(ValueError, match="ligand or pocket mask is empty"):
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=atom_array,
            settings=PocketSamplingSettings(rdkit_num_conformers=0),
        )


def test_create_pocket_sampling_features_uses_carbon_radius_for_unknown_elements():
    atom_array = _atom_array()
    atom_array.element[0] = "Xx"

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=atom_array,
        settings=PocketSamplingSettings(rdkit_num_conformers=0),
    )

    assert features["pocket_sampling_vdw_radii"][0].item() == pytest.approx(
        pocket_constraints.DEFAULT_VDW_RADIUS
    )


def test_create_pocket_sampling_features_uses_default_vdw_radius_for_non_string_element():
    atom_array = _atom_array()
    atom_array.element = atom_array.element.astype(object)
    atom_array.element[0] = None

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=atom_array,
        settings=PocketSamplingSettings(rdkit_num_conformers=0),
    )

    assert features["pocket_sampling_vdw_radii"][0].item() == pytest.approx(
        pocket_constraints.DEFAULT_VDW_RADIUS
    )


def test_resolve_ligand_reference_molecule_rejects_missing_sequence():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "smiles": "CCO",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="has no sequence"):
        pocket_constraints._resolve_ligand_reference_molecule(
            query=query,
            processed_reference_molecules=[],
            ligand_chain_id="L",
        )


def test_resolve_ligand_reference_molecule_rejects_short_reference_list():
    query = _query_with_pocket_constraint()

    with pytest.raises(ValueError, match="Not enough processed reference molecules"):
        pocket_constraints._resolve_ligand_reference_molecule(
            query=query,
            processed_reference_molecules=[],
            ligand_chain_id="L",
        )


def test_resolve_ligand_reference_molecule_returns_none_for_unknown_ligand_id():
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    assert (
        pocket_constraints._resolve_ligand_reference_molecule(
            query=query,
            processed_reference_molecules=structure.processed_reference_mols,
            ligand_chain_id="Z",
        )
        is None
    )


def test_resolve_ligand_reference_molecule_rejects_unsupported_molecule_type():
    class UnsupportedChain:
        molecule_type = "unsupported"
        chain_ids = ["A"]

    class UnsupportedQuery:
        chains = [UnsupportedChain()]

    with pytest.raises(ValueError, match="Unsupported molecule type"):
        pocket_constraints._resolve_ligand_reference_molecule(
            query=UnsupportedQuery(),
            processed_reference_molecules=[],
            ligand_chain_id="L",
        )


def test_create_pocket_sampling_features_skips_conformers_for_bad_atom_names():
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array.copy()
    ligand_indices = np.flatnonzero(atom_array.chain_id == "L")
    atom_array.atom_name[ligand_indices[0]] = "BAD"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_for_bad_atom_elements():
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array.copy()
    ligand_indices = np.flatnonzero(atom_array.chain_id == "L")
    atom_array.element[ligand_indices[0]] = "N"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_generates_conformers_from_reference_molecule():
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    lig_mask = structure.atom_array.chain_id == "L"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=2, rdkit_conformer_rng=17),
    )

    rels = features["pocket_sampling_conformer_rels"]
    assert rels.shape[0] >= 1
    assert rels.shape[1:] == (int(lig_mask.sum()), 3)
    assert torch.allclose(
        rels.mean(dim=1),
        torch.zeros(rels.shape[0], 3),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "chains",
    [
        [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "AC",
            },
            {
                "molecule_type": "ligand",
                "chain_ids": "X",
                "smiles": "N#N",
            },
            {
                "molecule_type": "ligand",
                "chain_ids": "L",
                "smiles": "CC(=O)O",
            },
        ],
        [
            {
                "molecule_type": "ligand",
                "chain_ids": "L",
                "smiles": "CC(=O)O",
            },
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "AC",
            },
            {
                "molecule_type": "ligand",
                "chain_ids": "X",
                "smiles": "N#N",
            },
        ],
    ],
)
def test_create_pocket_sampling_features_resolves_ligand_reference_by_query_order(
    chains,
):
    query = Query.model_validate(
        {
            "chains": chains,
            "pocket_constraint": {
                "ligand_chain_id": "L",
                "pocket_residues": [["A", 2]],
            },
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    ligand_atom_count = int((structure.atom_array.chain_id == "L").sum())

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert features["pocket_sampling_conformer_rels"].shape[1:] == (
        ligand_atom_count,
        3,
    )


def test_create_pocket_sampling_features_generates_conformers_from_ccd_reference_molecule():
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "AC",
                },
                {
                    "molecule_type": "ligand",
                    "chain_ids": "L",
                    "ccd_codes": ["EOH"],
                },
            ],
            "pocket_constraint": {
                "ligand_chain_id": "L",
                "pocket_residues": [["A", 2]],
            },
        }
    )
    structure = structure_with_ref_mols_from_query(query)
    lig_mask = structure.atom_array.chain_id == "L"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=2, rdkit_conformer_rng=17),
    )

    rels = features["pocket_sampling_conformer_rels"]
    assert rels.shape[0] >= 1
    assert rels.shape[1:] == (int(lig_mask.sum()), 3)


def test_create_pocket_sampling_features_skips_conformers_without_reference_molecules():
    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=_atom_array(),
        settings=PocketSamplingSettings(rdkit_num_conformers=2),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_when_ligand_ref_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        pocket_constraints,
        "_resolve_ligand_reference_molecule",
        lambda query, processed_reference_molecules, ligand_chain_id: None,
    )
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_for_hydrogen_atom_order(
    monkeypatch,
):
    monkeypatch.setattr(
        pocket_constraints,
        "_atom_order_from_reference_molecule",
        lambda processed_reference_molecule, ligand_atom_array: [
            processed_reference_molecule.mol.GetNumAtoms()
        ],
    )
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_can_use_uff_conformer_optimization(
    monkeypatch,
):
    from rdkit.Chem import AllChem

    monkeypatch.setattr(AllChem, "MMFFHasAllMoleculeParams", lambda _mol: False)
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert features["pocket_sampling_conformer_rels"].shape[0] == 1


def test_create_pocket_sampling_features_skips_conformers_on_generation_error(
    monkeypatch,
):
    monkeypatch.setattr(
        pocket_constraints,
        "_atom_order_from_reference_molecule",
        lambda processed_reference_molecule, ligand_atom_array: (_ for _ in ()).throw(
            ValueError("bad mapping")
        ),
    )
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
        settings=PocketSamplingSettings(rdkit_num_conformers=1),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_respects_disabled_setting():
    # Disabling pocket setting at the config level creates empty features
    assert (
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=_atom_array(),
            settings=PocketSamplingSettings(enabled=False),
        )
        == {}
    )


def test_create_noise_schedule_matches_expected_endpoints():
    schedule = create_noise_schedule(
        no_rollout_steps=2,
        sigma_data=1.0,
        s_max=4.0,
        s_min=1.0,
        p=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert schedule.tolist() == pytest.approx([4.0, 2.25, 1.0])


def test_feature_mask_normalizes_singleton_dimensions():
    batch = {"mask": torch.tensor([[[1, 0, 1]]])}
    atom_mask = torch.ones(2, 3)

    mask = _feature_mask(batch, atom_mask, "mask")

    assert mask.shape == (2, 3)
    assert mask.tolist() == [[True, False, True], [True, False, True]]


def test_feature_mask_normalizes_one_dimensional_mask():
    batch = {"mask": torch.tensor([1, 0, 1])}
    atom_mask = torch.ones(1, 3)

    mask = _feature_mask(batch, atom_mask, "mask")

    assert mask.shape == (1, 3)
    assert mask.tolist() == [[True, False, True]]


class _IdentityDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, xl_noisy, **_kwargs):
        self.calls += 1
        return xl_noisy


def _pocket_sampling_batch_without_jitter(
    batch_dim: int = 1,
) -> dict[str, torch.Tensor]:
    return {
        "atom_mask": torch.ones(batch_dim, 5),
        "token_mask": torch.ones(batch_dim, 1),
        "pocket_sampling_enabled": torch.tensor([True]),
        "pocket_sampling_ligand_atom_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "pocket_sampling_pocket_atom_mask": torch.tensor([[1, 1, 0, 0, 0]]),
        "pocket_sampling_vdw_radii": torch.full((5,), 1.7),
        "pocket_sampling_contact_distance": torch.tensor([4.0]),
        "pocket_sampling_num_parents": torch.tensor([2]),
        "pocket_sampling_candidates": torch.tensor([2]),
        "pocket_sampling_start_frac": torch.tensor([0.5]),
        "pocket_sampling_ligand_jitter": torch.tensor([0.0]),
        "pocket_sampling_center_jitter": torch.tensor([0.0]),
        "pocket_sampling_surface_jitter": torch.tensor([0.0]),
        "pocket_sampling_vdw_buffer": torch.tensor([0.0]),
        "pocket_sampling_diversity_rmsd": torch.tensor([0.0]),
    }


def test_build_pocket_sampling_seeds_uses_generated_conformer_candidates():
    torch.manual_seed(0)
    batch = _pocket_sampling_batch_without_jitter()
    batch["pocket_sampling_candidates"] = torch.tensor([6])
    input_conformer = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch["pocket_sampling_conformer_rels"] = input_conformer[None, None, None]

    xl_base = torch.zeros(1, 2, 5, 3)
    xl_base[:, :, 0] = torch.tensor([0.0, 0.0, 0.0])
    xl_base[:, :, 1] = torch.tensor([0.0, 0.0, 0.0])
    xl_base[:, :, 2] = torch.tensor([20.0, 0.0, 0.0])
    xl_base[:, :, 3] = torch.tensor([10.0, 0.0, 0.0])
    xl_base[:, :, 4] = torch.tensor([11.0, 0.0, 0.0])

    seeds = _build_pocket_sampling_seeds(
        batch=batch,
        xl_base=xl_base,
        atom_mask=batch["atom_mask"],
        no_rollout_samples=2,
    )

    ligand = seeds[0, :, 3:5]
    ligand_com = ligand.mean(dim=1)
    ligand_distance = torch.linalg.vector_norm(ligand[:, 0] - ligand[:, 1], dim=-1)
    expected_distance = torch.linalg.vector_norm(
        input_conformer[0] - input_conformer[1]
    )

    assert torch.all(torch.linalg.vector_norm(ligand_com, dim=-1) < 1e-5), (
        "ligand COM should be at origin"
    )
    assert torch.allclose(
        ligand_distance,
        expected_distance.expand_as(ligand_distance),
        atol=1e-5,
    ), "rigid placement should preserve the input conformer's interatomic distance"


def test_build_pocket_sampling_seeds_uses_parent_conformer_and_soft_overlap_score():
    torch.manual_seed(1)
    batch = _pocket_sampling_batch_without_jitter()
    batch["pocket_sampling_num_parents"] = torch.tensor([1])
    batch["pocket_sampling_candidates"] = torch.tensor([4])
    batch["pocket_sampling_diversity_rmsd"] = torch.tensor([999.0])
    batch["pocket_sampling_vdw_radii"] = torch.full((1, 1, 5), 1.7)

    xl_base = torch.zeros(1, 2, 5, 3)
    xl_base[:, :, 0] = torch.tensor([0.0, 0.0, 0.0])
    xl_base[:, :, 1] = torch.tensor([0.0, 0.0, 0.0])
    xl_base[:, :, 2] = torch.tensor([0.0, 0.0, 0.0])
    xl_base[:, :, 3] = torch.tensor([0.1, 0.0, 0.0])
    xl_base[:, :, 4] = torch.tensor([0.2, 0.0, 0.0])

    seeds = _build_pocket_sampling_seeds(
        batch=batch,
        xl_base=xl_base,
        atom_mask=batch["atom_mask"],
        no_rollout_samples=2,
    )

    assert seeds.shape == xl_base.shape, (
        "seed generation should preserve the parent batch, rollout, atom, "
        "and coordinate dimensions"
    )


def test_sample_diffusion_runs_second_pass_when_pocket_sampling_enabled():
    denoiser = _IdentityDenoiser()
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=denoiser,
    )

    with torch.no_grad():
        result = sampler(
            batch=_pocket_sampling_batch_without_jitter(),
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )

    assert result.shape == (1, 2, 5, 3)
    assert denoiser.calls == 3


def test_sample_diffusion_applies_independent_rigid_ligand_jitter(monkeypatch):
    batch = _pocket_sampling_batch_without_jitter()
    jitter_scale = 0.25
    batch["pocket_sampling_ligand_jitter"] = torch.tensor([jitter_scale])
    seed = torch.arange(30, dtype=torch.float32).reshape(1, 2, 5, 3)
    jitter_draw = torch.tensor([[[[1.0, -2.0, 0.5]], [[-0.5, 1.0, 2.0]]]])

    def controlled_randn(shape, *, device=None, dtype=None):
        if tuple(shape) == tuple(jitter_draw.shape):
            return jitter_draw.to(device=device, dtype=dtype)
        return torch.zeros(shape, device=device, dtype=dtype)

    monkeypatch.setattr(diffusion_module.torch, "randn", controlled_randn)
    monkeypatch.setattr(
        diffusion_module.torch,
        "randn_like",
        lambda value: torch.zeros_like(value),
    )
    monkeypatch.setattr(
        diffusion_module,
        "centre_random_augmentation",
        lambda *, xl, atom_mask: xl,
    )
    monkeypatch.setattr(
        diffusion_module,
        "_build_pocket_sampling_seeds",
        lambda **_kwargs: seed.clone(),
    )

    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )
    with torch.no_grad():
        result = sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )

    expected_shift = jitter_scale * jitter_draw
    ligand_shift = result[:, :, 3:] - seed[:, :, 3:]
    assert torch.allclose(result[:, :, :3], seed[:, :, :3]), (
        "ligand jitter should not move protein atoms"
    )
    assert torch.allclose(ligand_shift, expected_shift.expand_as(ligand_shift)), (
        "each rollout should apply one independently sampled rigid ligand translation"
    )
    assert torch.allclose(
        torch.linalg.vector_norm(result[:, :, 3] - result[:, :, 4], dim=-1),
        torch.linalg.vector_norm(seed[:, :, 3] - seed[:, :, 4], dim=-1),
    ), "rigid ligand jitter should preserve intraligand distances"


def test_sample_diffusion_requires_complete_pocket_sampling_features():
    """Pocket sampling features are assumed complete (see pocket_constraints.py

    docstring); a malformed batch fails with a plain KeyError once the missing
    feature is actually read, not via upfront validation.
    """
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )
    batch = _pocket_sampling_batch_without_jitter()
    del batch["pocket_sampling_vdw_buffer"]

    with pytest.raises(KeyError, match="pocket_sampling_vdw_buffer"):
        sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )
    # The first (unconstrained) rollout already ran before the missing
    # feature was discovered while building the pocket-sampling seeds.
    assert sampler.diffusion_module.calls == 2


def test_sample_diffusion_rejects_multi_query_pocket_sampling_batch():
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )

    with pytest.raises(ValueError, match="one query per model batch"):
        sampler(
            batch=_pocket_sampling_batch_without_jitter(batch_dim=2),
            si_input=torch.zeros(2, 1, 1),
            si_trunk=torch.zeros(2, 1, 1),
            zij_trunk=torch.zeros(2, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )
