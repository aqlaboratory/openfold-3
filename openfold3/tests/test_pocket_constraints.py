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

from copy import deepcopy

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from pydantic import ValidationError
from rdkit import Chem

from openfold3.core.data.pipelines.featurization.pocket_constraints import (
    create_pocket_sampling_features,
    read_bool_env,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mol_from_smiles,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.model.structure.diffusion_module import SampleDiffusion
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
            "pocket_constraints": [
                {
                    "ligand_chain_id": "L",
                    "pocket_residues": [["A", 2]],
                    "max_distance": 3.5,
                }
            ],
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

    constraint = query.pocket_constraints[0]
    assert constraint.ligand_chain_id == "L"
    assert constraint.pocket_residues[0].chain_id == "A"
    assert constraint.pocket_residues[0].residue_id == 2
    assert constraint.max_distance == 3.5


def test_pocket_constraints_reject_unused_strength_field():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraints"][0]["strength"] = 1.0

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
    payload["pocket_constraints"][0].update(update)

    with pytest.raises(ValidationError, match=match):
        Query.model_validate(payload)


def test_pocket_constraints_reject_multiple_constraints():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraints"].append(deepcopy(payload["pocket_constraints"][0]))

    with pytest.raises(
        ValidationError, match="Exactly one pocket constraint is currently supported"
    ):
        Query.model_validate(payload)


def test_pocket_constraints_ligand_chain_must_reference_ligand():
    payload = _query_with_pocket_constraint().model_dump()
    payload["pocket_constraints"][0]["ligand_chain_id"] = "A"

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
    assert features["pocket_sampling_num_parents"].item() == 16
    assert features["pocket_sampling_candidates"].item() == 1024
    assert features["pocket_sampling_start_frac"].item() == pytest.approx(0.75)
    assert features["pocket_sampling_ligand_jitter"].item() == pytest.approx(0.25)
    assert features["pocket_sampling_translate"].item() == pytest.approx(0.0)
    assert "pocket_sampling_conformer_rels" not in features


def test_smiles_ligand_atom_order_matches_rdkit_heavy_atom_order():
    query = _query_with_pocket_constraint()
    smiles = query.chains[1].smiles
    structure = structure_with_ref_mol_from_smiles(
        smiles=smiles, chain_id="L", res_name="LIG0"
    )
    atom_array = structure.atom_array
    rdkit_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdkit_elements = [
        atom.GetSymbol().upper()
        for atom in rdkit_mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    assert [str(element).upper() for element in atom_array.element] == rdkit_elements


def test_create_pocket_sampling_features_generates_rdkit_conformers_from_smiles(
    monkeypatch,
):
    monkeypatch.setenv("OF3_POCKET_SAMPLING_NUM_CONFORMERS", "2")
    monkeypatch.setenv("OF3_POCKET_SAMPLING_CONFORMER_RNG", "17")

    features = create_pocket_sampling_features(
        query=_query_with_pocket_constraint(),
        atom_array=_atom_array(),
    )

    rels = features["pocket_sampling_conformer_rels"]
    assert rels.shape[0] >= 1
    assert rels.shape[1:] == (3, 3)
    assert torch.allclose(
        rels.mean(dim=1),
        torch.zeros(rels.shape[0], 3),
        atol=1e-5,
    )


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

    with pytest.raises(ValueError, match=match):
        create_pocket_sampling_features(
            query=_query_with_pocket_constraint(),
            atom_array=_atom_array(),
        )


def test_read_bool_env_accepts_expected_values(monkeypatch):
    monkeypatch.delenv("OF3_POCKET_SAMPLING", raising=False)
    assert read_bool_env("OF3_POCKET_SAMPLING", default=True) is True

    monkeypatch.setenv("OF3_POCKET_SAMPLING", "off")
    assert read_bool_env("OF3_POCKET_SAMPLING", default=True) is False

    monkeypatch.setenv("OF3_POCKET_SAMPLING", "YES")
    assert read_bool_env("OF3_POCKET_SAMPLING", default=False) is True


class _IdentityDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, xl_noisy, **_kwargs):
        self.calls += 1
        return xl_noisy


def test_sample_diffusion_runs_second_pass_when_pocket_sampling_enabled():
    denoiser = _IdentityDenoiser()
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=denoiser,
    )
    batch = {
        "atom_mask": torch.ones(1, 5),
        "token_mask": torch.ones(1, 1),
        "pocket_sampling_enabled": torch.tensor([True]),
        "pocket_sampling_ligand_atom_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "pocket_sampling_pocket_atom_mask": torch.tensor([[1, 1, 0, 0, 0]]),
        "pocket_sampling_vdw_radii": torch.full((5,), 1.7),
        "pocket_sampling_contact_distance": torch.tensor([4.0]),
        "pocket_sampling_num_parents": torch.tensor([2]),
        "pocket_sampling_candidates": torch.tensor([2]),
        "pocket_sampling_start_frac": torch.tensor([0.5]),
        "pocket_sampling_ligand_jitter": torch.tensor([0.0]),
        "pocket_sampling_diversity_rmsd": torch.tensor([0.0]),
    }

    with torch.no_grad():
        result = sampler(
            batch=batch,
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )

    assert result.shape == (1, 2, 5, 3)
    assert denoiser.calls == 3


def test_sample_diffusion_rejects_multi_query_pocket_sampling_batch():
    sampler = SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        diffusion_module=_IdentityDenoiser(),
    )
    batch = {
        "atom_mask": torch.ones(2, 5),
        "token_mask": torch.ones(2, 1),
        "pocket_sampling_enabled": torch.tensor([True]),
        "pocket_sampling_ligand_atom_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "pocket_sampling_pocket_atom_mask": torch.tensor([[1, 1, 0, 0, 0]]),
    }

    with pytest.raises(ValueError, match="one query per model batch"):
        sampler(
            batch=batch,
            si_input=torch.zeros(2, 1, 1),
            si_trunk=torch.zeros(2, 1, 1),
            zij_trunk=torch.zeros(2, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5, 0.1]),
            no_rollout_samples=2,
        )
