import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from pydantic import ValidationError

from openfold3.core.config import pocket_sampling_defaults as defaults
from openfold3.core.data.pipelines.featurization import pocket_constraints
from openfold3.core.data.pipelines.featurization.pocket_constraints import (
    create_pocket_sampling_features,
    read_bool_env,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.resources.residues import MoleculeType
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
    atom_array = AtomArray(7)
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
        defaults.DEFAULT_POCKET_CONSTRAINT_MAX_DISTANCE
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


def test_create_pocket_sampling_features_uses_defaults(monkeypatch):
    monkeypatch.delenv("OF3_POCKET_SAMPLING", raising=False)
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "0")

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=_atom_array(),
    )

    assert features["pocket_sampling_enabled"].item() is True
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
    assert (
        features["pocket_sampling_num_parents"].item()
        == defaults.DEFAULT_POCKET_SAMPLING_NUM_PARENTS
    )
    assert (
        features["pocket_sampling_candidates"].item()
        == defaults.DEFAULT_POCKET_SAMPLING_CANDIDATES
    )
    assert features["pocket_sampling_start_frac"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_NOISE_FRAC
    )
    assert features["pocket_sampling_ligand_jitter"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_LIGAND_JITTER
    )
    assert features["pocket_sampling_center_jitter"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_CENTER_JITTER
    )
    assert features["pocket_sampling_surface_jitter"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_SURFACE_JITTER
    )
    assert features["pocket_sampling_vdw_buffer"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_VDW_BUFFER
    )
    assert features["pocket_sampling_diversity_rmsd"].item() == pytest.approx(
        defaults.DEFAULT_POCKET_SAMPLING_DIVERSITY_RMSD
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


def test_create_pocket_sampling_features_rejects_missing_pocket_residue(monkeypatch):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "0")
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraint"]["pocket_residues"].append(("A", 99))

    with pytest.raises(ValueError, match="A:99"):
        create_pocket_sampling_features(
            query=Query.model_validate(payload),
            atom_array=_atom_array(),
        )


def test_create_pocket_sampling_features_rejects_empty_ligand_mask(monkeypatch):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "0")
    atom_array = _atom_array()
    atom_array.molecule_type_id[atom_array.chain_id == "L"] = MoleculeType.PROTEIN

    with pytest.raises(ValueError, match="ligand or pocket mask is empty"):
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=atom_array,
        )


def test_create_pocket_sampling_features_uses_carbon_radius_for_unknown_elements(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "0")
    atom_array = _atom_array()
    atom_array.element[0] = "Xx"

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=atom_array,
    )

    assert features["pocket_sampling_vdw_radii"][0].item() == pytest.approx(
        pocket_constraints.DEFAULT_VDW_RADIUS
    )


def test_create_pocket_sampling_features_uses_default_vdw_radius_for_non_string_element(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "0")
    atom_array = _atom_array()
    atom_array.element = atom_array.element.astype(object)
    atom_array.element[0] = None

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=atom_array,
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


def test_create_pocket_sampling_features_skips_conformers_for_bad_atom_names(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array.copy()
    ligand_indices = np.flatnonzero(atom_array.chain_id == "L")
    atom_array.atom_name[ligand_indices[0]] = "BAD"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_for_bad_atom_elements(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    atom_array = structure.atom_array.copy()
    ligand_indices = np.flatnonzero(atom_array.chain_id == "L")
    atom_array.element[ligand_indices[0]] = "N"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_generates_conformers_from_reference_molecule(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "2")
    monkeypatch.setenv("OF3_POCKET_SAMPLING_CONFORMER_RNG", "17")
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)
    lig_mask = structure.atom_array.chain_id == "L"

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
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
    monkeypatch, chains
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
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
    )

    assert features["pocket_sampling_conformer_rels"].shape[1:] == (
        ligand_atom_count,
        3,
    )


def test_create_pocket_sampling_features_generates_conformers_from_ccd_reference_molecule(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "2")
    monkeypatch.setenv("OF3_POCKET_SAMPLING_CONFORMER_RNG", "17")
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
    )

    rels = features["pocket_sampling_conformer_rels"]
    assert rels.shape[0] >= 1
    assert rels.shape[1:] == (int(lig_mask.sum()), 3)


def test_create_pocket_sampling_features_skips_conformers_without_reference_molecules(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "2")

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=_atom_array(),
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_when_ligand_ref_is_missing(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
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
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_skips_conformers_for_hydrogen_atom_order(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
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
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_can_use_uff_conformer_optimization(
    monkeypatch,
):
    from rdkit.Chem import AllChem

    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
    monkeypatch.setattr(AllChem, "MMFFHasAllMoleculeParams", lambda _mol: False)
    query = _query_with_pocket_constraint()
    structure = structure_with_ref_mols_from_query(query)

    features = create_pocket_sampling_features(
        query=query,
        atom_array=structure.atom_array,
        processed_reference_molecules=structure.processed_reference_mols,
    )

    assert features["pocket_sampling_conformer_rels"].shape[0] == 1


def test_create_pocket_sampling_features_skips_conformers_on_generation_error(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "1")
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
    )

    assert "pocket_sampling_conformer_rels" not in features


def test_create_pocket_sampling_features_respects_disable_env(monkeypatch):
    monkeypatch.setenv("OF3_POCKET_SAMPLING", "0")

    assert (
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=_atom_array(),
        )
        == {}
    )


def test_create_pocket_sampling_features_validates_boolean_env(monkeypatch):
    monkeypatch.setenv("OF3_POCKET_SAMPLING", "maybe")

    with pytest.raises(ValueError, match="OF3_POCKET_SAMPLING must be one of"):
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=_atom_array(),
        )


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        (
            "OF3_POCKET_SAMPLING_NUM_PARENTS",
            "zero",
            "OF3_POCKET_SAMPLING_NUM_PARENTS must be an integer",
        ),
        (
            "OF3_POCKET_SAMPLING_NOISE_FRAC",
            "1.5",
            "OF3_POCKET_SAMPLING_NOISE_FRAC must be <= 1.0",
        ),
        (
            "OF3_POCKET_SAMPLING_LIGAND_JITTER",
            "-1",
            "OF3_POCKET_SAMPLING_LIGAND_JITTER must be >= 0.0",
        ),
        (
            "OF3_POCKET_SAMPLING_CENTER_JITTER",
            "not-a-float",
            "OF3_POCKET_SAMPLING_CENTER_JITTER must be a finite float",
        ),
        (
            "OF3_POCKET_SAMPLING_SURFACE_JITTER",
            "nan",
            "OF3_POCKET_SAMPLING_SURFACE_JITTER must be a finite float",
        ),
        (
            "OF3_POCKET_SAMPLING_CONFORMER_MAX_ITERS",
            "0",
            "OF3_POCKET_SAMPLING_CONFORMER_MAX_ITERS must be >= 1",
        ),
    ],
)
def test_create_pocket_sampling_features_validates_numeric_env(
    monkeypatch, name, value, match
):
    monkeypatch.setenv(name, value)
    query = _query_with_pocket_constraint()
    kwargs = {
        "query": query,
        "atom_array": _atom_array(),
    }
    if name.startswith("OF3_POCKET_SAMPLING_CONFORMER"):
        structure = structure_with_ref_mols_from_query(query)
        kwargs = {
            "query": query,
            "atom_array": structure.atom_array,
            "processed_reference_molecules": structure.processed_reference_mols,
        }

    with pytest.raises(ValueError, match=match):
        create_pocket_sampling_features(**kwargs)


def test_read_bool_env_accepts_expected_values(monkeypatch):
    monkeypatch.delenv("OF3_POCKET_SAMPLING", raising=False)
    assert (
        read_bool_env(
            "OF3_POCKET_SAMPLING",
            default=defaults.DEFAULT_POCKET_SAMPLING_ENABLED,
        )
        is True
    )

    monkeypatch.setenv("OF3_POCKET_SAMPLING", "off")
    assert read_bool_env("OF3_POCKET_SAMPLING", default=True) is False

    monkeypatch.setenv("OF3_POCKET_SAMPLING", "YES")
    assert read_bool_env("OF3_POCKET_SAMPLING", default=False) is True


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


def _pocket_sampling_batch(batch_dim: int = 1) -> dict[str, torch.Tensor]:
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
    batch = _pocket_sampling_batch()
    batch["pocket_sampling_candidates"] = torch.tensor([6])
    batch["pocket_sampling_conformer_rels"] = torch.tensor(
        [[[[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]]]
    )

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

    assert torch.all(torch.linalg.vector_norm(ligand_com, dim=-1) < 1e-5)
    assert torch.allclose(ligand_distance, torch.full((2,), 4.0), atol=1e-5)


def test_build_pocket_sampling_seeds_uses_parent_conformer_and_soft_overlap_score():
    torch.manual_seed(1)
    batch = _pocket_sampling_batch()
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

    assert seeds.shape == (1, 2, 5, 3)


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
            batch=_pocket_sampling_batch(),
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )

    assert result.shape == (1, 2, 5, 3)
    assert denoiser.calls == 3


def test_sample_diffusion_requires_complete_pocket_sampling_features():
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )
    batch = _pocket_sampling_batch()
    del batch["pocket_sampling_vdw_buffer"]

    with pytest.raises(ValueError, match="pocket_sampling_vdw_buffer"):
        sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )
    assert sampler.diffusion_module.calls == 0


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
            batch=_pocket_sampling_batch(batch_dim=2),
            si_input=torch.zeros(2, 1, 1),
            si_trunk=torch.zeros(2, 1, 1),
            zij_trunk=torch.zeros(2, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )
