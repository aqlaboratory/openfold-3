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

"""Shared structure builders for the steering tests.

Not named ``test_*`` so pytest does not collect it.
"""

from __future__ import annotations

import numpy as np
import torch

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    tokenize_atom_array,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

PROTEIN_A = {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GAGA"}
PROTEIN_B = {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "AAGG"}
LIGAND_L = {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "C[C@H](O)C(=O)O"}
LIGAND_M = {"molecule_type": "ligand", "chain_ids": ["M"], "smiles": "c1ccccc1"}

# Layouts that renumber the global atom axis relative to a ligand-only query.
LAYOUTS: dict[str, list[dict]] = {
    "ligand_after_protein": [PROTEIN_A, LIGAND_L],
    "ligand_before_protein": [LIGAND_L, PROTEIN_A],
    "ligand_between_proteins": [PROTEIN_A, LIGAND_L, PROTEIN_B],
    "two_ligands": [PROTEIN_A, LIGAND_L, LIGAND_M],
    "repeated_protein_chains": [
        {"molecule_type": "protein", "chain_ids": ["A", "B"], "sequence": "GAGA"},
        LIGAND_L,
    ],
}


def structure_for(chains: list[dict]):
    """Build a tokenized structure with reference molecules from chain specs."""
    structure = structure_with_ref_mols_from_query(
        Query.model_validate({"chains": chains})
    )
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    return structure


def ligand_atom_indices(structure) -> np.ndarray:
    return np.flatnonzero(structure.atom_array.molecule_type_id == MoleculeType.LIGAND)


def reference_coords_by_atom_name(structure) -> torch.Tensor:
    """Place each ligand's reference conformer on the global atom axis.

    Maps by atom name rather than by position, so this is independent of the
    positional ``in_crop_mask`` zip that ``build_context`` uses -- which is
    what makes it usable as an oracle for that mapping.
    """
    atom_array = structure.atom_array
    coords = torch.zeros((len(atom_array), 3), dtype=torch.float32)
    global_indices = np.arange(len(atom_array))
    for residue, reference in zip(
        residue_view_iter(atom_array), structure.processed_reference_mols, strict=True
    ):
        if not np.all(residue.molecule_type_id == MoleculeType.LIGAND):
            continue
        conformer = reference.mol.GetConformer()
        local_by_name = {
            atom.GetProp("annot_atom_name"): atom.GetIdx()
            for atom in reference.mol.GetAtoms()
        }
        for global_index in global_indices[residue.indices]:
            name = str(atom_array.atom_name[global_index])
            position = conformer.GetAtomPosition(local_by_name[name])
            coords[global_index] = torch.tensor(
                [position.x, position.y, position.z], dtype=torch.float32
            )
    return coords


def collate(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Put features through the real collator, then the sample-axis unsqueeze.

    Uses `openfold_batch_collator` itself rather than a stand-in, so these
    tests cannot drift from what the dataloader actually does to a feature
    dict. The `unsqueeze(1)` afterwards mirrors the `tensor_tree_map` in
    projects/of3_all_atom/model.py, which is the other half of what a feature
    experiences before the sampler sees it.
    """
    batch = openfold_batch_collator([dict(features)])
    return {key: value.unsqueeze(1) for key, value in batch.items()}
