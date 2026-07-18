"""Featurization helpers for query-level ligand pocket constraints."""

import logging
import os

import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.core.config import pocket_sampling_defaults as defaults
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.labels import uniquify_ids
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
DEFAULT_VDW_RADIUS = VDW_RADII["C"]


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


def _resolve_ligand_reference_molecule(
    query: Query,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
    ligand_chain_id: str,
) -> ProcessedReferenceMolecule | None:
    """Find the processed reference molecule matching a query ligand chain."""
    ref_mol_idx = 0
    for chain in query.chains:
        for chain_id in chain.chain_ids:
            match chain.molecule_type:
                case MoleculeType.PROTEIN | MoleculeType.DNA | MoleculeType.RNA:
                    if chain.sequence is None:
                        raise ValueError(
                            f"Chain {chain_id} has no sequence but is "
                            "required to resolve pocket sampling reference molecules"
                        )
                    ref_mol_idx += len(chain.sequence)
                case MoleculeType.LIGAND:
                    if ref_mol_idx >= len(processed_reference_molecules):
                        raise ValueError(
                            "Not enough processed reference molecules to resolve "
                            f"ligand chain {ligand_chain_id!r}"
                        )
                    if chain_id == ligand_chain_id:
                        return processed_reference_molecules[ref_mol_idx]
                    ref_mol_idx += 1
                case _:
                    raise ValueError(
                        f"Unsupported molecule type: {chain.molecule_type}"
                    )
    return None


def _atom_order_from_reference_molecule(
    processed_reference_molecule: ProcessedReferenceMolecule,
    ligand_atom_array: AtomArray,
) -> list[int]:
    """Map OF3 ligand atom order onto annotated reference-molecule atom indices."""
    mol = processed_reference_molecule.mol
    atom_names = np.asarray(
        [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()], dtype=object
    )
    atom_elements = np.asarray(
        [atom.GetSymbol().upper() for atom in mol.GetAtoms()], dtype=object
    )
    in_crop_mask = np.asarray(processed_reference_molecule.in_crop_mask, dtype=bool)

    ref_keys = np.asarray(uniquify_ids(atom_names[in_crop_mask].tolist()))
    ligand_keys = np.asarray(uniquify_ids(ligand_atom_array.atom_name.tolist()))

    if len(ref_keys) != len(ligand_keys) or set(ref_keys) != set(ligand_keys):
        raise ValueError(
            "Reference molecule atom names do not match OF3 ligand atom names"
        )

    ref_indices_by_key = {
        key: int(idx) for key, idx in zip(ref_keys, np.flatnonzero(in_crop_mask))
    }
    ordered_indices = [ref_indices_by_key[key] for key in ligand_keys]

    ref_ordered_elements = atom_elements[ordered_indices]
    ligand_elements = np.asarray(
        [str(element).upper() for element in ligand_atom_array.element], dtype=object
    )
    if not np.array_equal(ref_ordered_elements, ligand_elements):
        raise ValueError(
            "Reference molecule atom elements do not match OF3 ligand atom order"
        )

    return ordered_indices


def create_pocket_sampling_features(
    query: Query,
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule] | None = None,
) -> dict[str, torch.Tensor]:
    """Create sampler features for pocket proposal and partial-diffusion refinement.

    Runs automatically when a query provides pocket_constraint. The sampler
    receives the ligand mask, user-pocket atom mask, VDW radii, and optional
    RDKit ligand conformers from the OF3 reference molecule. OF3_POCKET_SAMPLING remains
    available as an explicit boolean override.
    """
    if query.pocket_constraint is None:
        return {}
    if not read_bool_env(
        "OF3_POCKET_SAMPLING", default=defaults.DEFAULT_POCKET_SAMPLING_ENABLED
    ):
        return {}

    constraint = query.pocket_constraint
    lig_mask = (atom_array.chain_id == constraint.ligand_chain_id) & (
        atom_array.molecule_type_id == MoleculeType.LIGAND
    )
    pocket_mask = np.zeros(len(atom_array), dtype=bool)
    for residue in constraint.pocket_residues:
        residue_mask = (atom_array.chain_id == residue.chain_id) & (
            atom_array.res_id == residue.residue_id
        )
        if not residue_mask.any():
            raise ValueError(
                "Pocket constraint residue "
                f"{residue.chain_id}:{residue.residue_id} does not match any atoms"
            )
        pocket_mask |= residue_mask
    if not lig_mask.any() or not pocket_mask.any():
        raise ValueError(
            "OF3_POCKET_SAMPLING requested but ligand or pocket mask is empty"
        )

    def _vdw_radius(element: object) -> float:
        if not isinstance(element, str):
            return DEFAULT_VDW_RADIUS
        return VDW_RADII.get(element.upper(), DEFAULT_VDW_RADIUS)

    features = {
        "pocket_sampling_enabled": torch.tensor([True], dtype=torch.bool),
        "pocket_sampling_ligand_atom_mask": torch.from_numpy(
            lig_mask.astype(np.float32)
        ),
        "pocket_sampling_pocket_atom_mask": torch.from_numpy(
            pocket_mask.astype(np.float32)
        ),
        "pocket_sampling_vdw_radii": torch.tensor(
            [_vdw_radius(e) for e in atom_array.element],
            dtype=torch.float32,
        ),
        "pocket_sampling_contact_distance": torch.tensor(
            [float(constraint.max_distance)], dtype=torch.float32
        ),
        "pocket_sampling_num_parents": torch.tensor(
            [
                read_int_env(
                    "OF3_POCKET_SAMPLING_NUM_PARENTS",
                    defaults.DEFAULT_POCKET_SAMPLING_NUM_PARENTS,
                    min_value=1,
                )
            ],
            dtype=torch.long,
        ),
        "pocket_sampling_candidates": torch.tensor(
            [
                read_int_env(
                    "OF3_POCKET_SAMPLING_CANDIDATES",
                    defaults.DEFAULT_POCKET_SAMPLING_CANDIDATES,
                    min_value=1,
                )
            ],
            dtype=torch.long,
        ),
        "pocket_sampling_start_frac": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_NOISE_FRAC",
                    defaults.DEFAULT_POCKET_SAMPLING_NOISE_FRAC,
                    min_value=0.0,
                    max_value=1.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_ligand_jitter": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_LIGAND_JITTER",
                    defaults.DEFAULT_POCKET_SAMPLING_LIGAND_JITTER,
                    min_value=0.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_center_jitter": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_CENTER_JITTER",
                    defaults.DEFAULT_POCKET_SAMPLING_CENTER_JITTER,
                    min_value=0.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_surface_jitter": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_SURFACE_JITTER",
                    defaults.DEFAULT_POCKET_SAMPLING_SURFACE_JITTER,
                    min_value=0.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_vdw_buffer": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_VDW_BUFFER",
                    defaults.DEFAULT_POCKET_SAMPLING_VDW_BUFFER,
                    min_value=0.0,
                )
            ],
            dtype=torch.float32,
        ),
        "pocket_sampling_diversity_rmsd": torch.tensor(
            [
                read_float_env(
                    "OF3_POCKET_SAMPLING_DIVERSITY_RMSD",
                    defaults.DEFAULT_POCKET_SAMPLING_DIVERSITY_RMSD,
                    min_value=0.0,
                )
            ],
            dtype=torch.float32,
        ),
    }

    n_conformers = read_int_env(
        "OF3_POCKET_SAMPLING_NUM_CONFORMERS",
        defaults.DEFAULT_POCKET_SAMPLING_NUM_CONFORMERS,
        min_value=0,
    )
    if n_conformers > 0 and processed_reference_molecules is not None:
        conformer_rng = read_int_env(
            "OF3_POCKET_SAMPLING_CONFORMER_RNG",
            defaults.DEFAULT_POCKET_SAMPLING_CONFORMER_RNG,
        )
        conformer_prune_rmsd = read_float_env(
            "OF3_POCKET_SAMPLING_CONFORMER_PRUNE_RMSD",
            defaults.DEFAULT_POCKET_SAMPLING_CONFORMER_PRUNE_RMSD,
            min_value=0.0,
        )
        conformer_max_iters = read_int_env(
            "OF3_POCKET_SAMPLING_CONFORMER_MAX_ITERS",
            defaults.DEFAULT_POCKET_SAMPLING_CONFORMER_MAX_ITERS,
            min_value=1,
        )
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            processed_ligand_mol = _resolve_ligand_reference_molecule(
                query=query,
                processed_reference_molecules=processed_reference_molecules,
                ligand_chain_id=constraint.ligand_chain_id,
            )
            if processed_ligand_mol is None:
                raise ValueError(
                    f"No processed reference molecule found for ligand chain "
                    f"{constraint.ligand_chain_id!r}"
                )
            ligand_atom_array = atom_array[lig_mask]
            heavy_indices = _atom_order_from_reference_molecule(
                processed_reference_molecule=processed_ligand_mol,
                ligand_atom_array=ligand_atom_array,
            )

            mol = Chem.Mol(processed_ligand_mol.mol)
            mol.RemoveAllConformers()
            mol_h = Chem.AddHs(mol)
            for atom_idx in heavy_indices:
                atom = mol_h.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() <= 1 or not atom.HasProp("annot_atom_name"):
                    raise ValueError(
                        "RDKit hydrogen expansion changed reference atom indices"
                    )
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
