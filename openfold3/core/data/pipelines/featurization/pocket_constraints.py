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

"""Featurization helpers for query-level ligand pocket constraints."""

import logging
import os

import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

logger = logging.getLogger(__name__)

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def read_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment override with explicit validation."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def read_int_env(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
) -> int:
    """Read an integer environment override with explicit validation."""
    value = os.environ.get(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {parsed}")
    return parsed


def read_float_env(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read a float environment override with explicit validation."""
    value = os.environ.get(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite float; got {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite float; got {parsed!r}")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {parsed}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}; got {parsed}")
    return parsed


def create_pocket_sampling_features(
    query: Query, atom_array: AtomArray
) -> dict[str, torch.Tensor]:
    """Create sampler features for pocket proposal and partial-diffusion refinement.

    Runs automatically when a query provides pocket_constraints. The sampler
    receives the ligand mask, user-pocket atom mask, VDW radii, and optional
    RDKit ligand conformers from the query SMILES. OF3_POCKET_SAMPLING remains
    available as an explicit boolean override.
    """
    if not query.pocket_constraints:
        return {}
    if not read_bool_env("OF3_POCKET_SAMPLING", default=True):
        return {}

    constraint = query.pocket_constraints[0]
    lig_mask = (atom_array.chain_id == constraint.ligand_chain_id) & (
        atom_array.molecule_type_id == MoleculeType.LIGAND
    )
    pocket_mask = np.zeros(len(atom_array), dtype=bool)
    for residue in constraint.pocket_residues:
        pocket_mask |= (atom_array.chain_id == residue.chain_id) & (
            atom_array.res_id == residue.residue_id
        )
    if not lig_mask.any() or not pocket_mask.any():
        raise ValueError(
            "OF3_POCKET_SAMPLING requested but ligand or pocket mask is empty"
        )

    def _vdw_radius(element: str) -> float:
        return VDW_RADII.get((element or "C").upper(), 1.70)

    def _ligand_smiles() -> str | None:
        for chain in query.chains:
            if constraint.ligand_chain_id in chain.chain_ids and chain.smiles:
                return chain.smiles
        return None

    features = {
        "pocket_sampling_enabled": torch.tensor([True], dtype=torch.bool),
        "pocket_sampling_ligand_atom_mask": torch.from_numpy(
            lig_mask.astype(np.float32)
        ),
        "pocket_sampling_pocket_atom_mask": torch.from_numpy(
            pocket_mask.astype(np.float32)
        ),
        "pocket_sampling_vdw_radii": torch.tensor(
            [_vdw_radius(str(e)) for e in atom_array.element],
            dtype=torch.float32,
        ),
        "pocket_sampling_contact_distance": torch.tensor(
            [float(constraint.max_distance)], dtype=torch.float32
        ),
        "pocket_sampling_num_parents": torch.tensor(
            [read_int_env("OF3_POCKET_SAMPLING_NUM_PARENTS", 16, min_value=1)],
            dtype=torch.long,
        ),
        "pocket_sampling_candidates": torch.tensor(
            [read_int_env("OF3_POCKET_SAMPLING_CANDIDATES", 1024, min_value=1)],
            dtype=torch.long,
        ),
        "pocket_sampling_start_frac": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_NOISE_FRAC",
                    0.75,
                    min_value=0.0,
                    max_value=1.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_ligand_jitter": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_LIGAND_JITTER", 0.25, min_value=0.0)],
            dtype=torch.float32,
        ),
        "pocket_sampling_translate": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_TRANSLATE", 0.0, min_value=0.0)],
            dtype=torch.float32,
        ),
        "pocket_sampling_center_jitter": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_CENTER_JITTER", 4.0, min_value=0.0)],
            dtype=torch.float32,
        ),
        "pocket_sampling_surface_jitter": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_SURFACE_JITTER", 1.5, min_value=0.0)],
            dtype=torch.float32,
        ),
        "pocket_sampling_vdw_buffer": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_VDW_BUFFER", 0.225, min_value=0.0)],
            dtype=torch.float32,
        ),
        "pocket_sampling_diversity_rmsd": torch.tensor(
            [read_float_env("OF3_POCKET_SAMPLING_DIVERSITY_RMSD", 0.5, min_value=0.0)],
            dtype=torch.float32,
        ),
    }

    n_conformers = read_int_env("OF3_POCKET_SAMPLING_NUM_CONFORMERS", 32, min_value=0)
    smiles = _ligand_smiles()
    if n_conformers > 0 and smiles is not None:
        conformer_rng = read_int_env("OF3_POCKET_SAMPLING_CONFORMER_RNG", 0)
        conformer_prune_rmsd = read_float_env(
            "OF3_POCKET_SAMPLING_CONFORMER_PRUNE_RMSD", 0.0, min_value=0.0
        )
        conformer_max_iters = read_int_env(
            "OF3_POCKET_SAMPLING_CONFORMER_MAX_ITERS", 200, min_value=1
        )
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit failed to parse ligand SMILES")
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = conformer_rng
            params.pruneRmsThresh = conformer_prune_rmsd
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=n_conformers,
                    params=params,
                )
            )
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                for conf_id in conf_ids:
                    AllChem.MMFFOptimizeMolecule(
                        mol_h, confId=int(conf_id), maxIters=conformer_max_iters
                    )
            else:
                for conf_id in conf_ids:
                    AllChem.UFFOptimizeMolecule(
                        mol_h, confId=int(conf_id), maxIters=conformer_max_iters
                    )

            heavy_indices = [
                atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1
            ]
            rdkit_elements = np.asarray(
                [mol_h.GetAtomWithIdx(i).GetSymbol().upper() for i in heavy_indices]
            )
            expected_elements = np.asarray(
                [str(e).upper() for e in atom_array.element[lig_mask]]
            )
            if len(rdkit_elements) != len(expected_elements) or not np.array_equal(
                rdkit_elements, expected_elements
            ):
                raise ValueError(
                    "RDKit conformer heavy-atom order does not match OF3 ligand order"
                )

            conformer_rels = []
            for conf_id in conf_ids:
                conf = mol_h.GetConformer(int(conf_id))
                conf_coords = np.asarray(
                    [
                        [
                            conf.GetAtomPosition(idx).x,
                            conf.GetAtomPosition(idx).y,
                            conf.GetAtomPosition(idx).z,
                        ]
                        for idx in heavy_indices
                    ],
                    dtype=np.float32,
                )
                conformer_rels.append(
                    conf_coords - conf_coords.mean(axis=0, keepdims=True)
                )
            if conformer_rels:
                features["pocket_sampling_conformer_rels"] = torch.from_numpy(
                    np.stack(conformer_rels, axis=0).astype(np.float32)
                )
            logger.info(
                "[pocket_sampling_build] rdkit_conformers=%s/%s",
                len(conformer_rels),
                n_conformers,
            )
        except Exception as exc:
            logger.warning(
                "[pocket_sampling_build] RDKit conformer generation failed; "
                "using parent ligand conformations only: %s: %s",
                type(exc).__name__,
                exc,
            )

    return features
