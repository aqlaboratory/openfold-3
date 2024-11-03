"""This module contains featurization pipelines for structural data."""

from typing import Union

import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.padding import pad_token_dim
from openfold3.core.data.primitives.featurization.structure import (
    create_sym_id,
    create_token_bonds,
    create_token_mask,
    encode_one_hot,
    extract_starts_entities,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_3,
    MoleculeType,
    get_with_unknown_3,
)


def featurize_structure_af3(
    atom_array: AtomArray,
    token_budget: int,
    token_dim_index_map: dict[str, int],
    is_gt: bool,
) -> dict[str, torch.Tensor]:
    """Creates target OR gt structure features following the AF3 strategy.

    Expects the cropped or duplicate-expanded AtomArray as input. Also pads tensors to
    crop size.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure.
        token_budget (int):
            Crop size.
        token_dim_index_map (dict[str, int]):
            Mapping of feature names to the index of the token dimension.
        is_gt (bool):
            Whether the input AtomArray is from the duplicate-expanded ground truth
            structure.

    Returns:
        dict[str, torch.Tensor]:
            Target or ground truth features.
    """
    token_starts_with_stop, entity_ids = extract_starts_entities(atom_array)
    token_starts = token_starts_with_stop[:-1]

    features = {}
    # Indexing
    features["residue_index"] = torch.tensor(
        atom_array.res_id[token_starts], dtype=torch.int32
    )
    features["token_index"] = torch.tensor(
        atom_array.token_id[token_starts], dtype=torch.int32
    )
    features["asym_id"] = torch.tensor(
        atom_array.chain_id[token_starts].astype(int), dtype=torch.int32
    )
    features["entity_id"] = torch.tensor(
        atom_array.entity_id[token_starts], dtype=torch.int32
    )
    features["sym_id"] = torch.tensor(
        create_sym_id(entity_ids, atom_array, token_starts), dtype=torch.int32
    )
    restype_index = torch.tensor(
        get_with_unknown_3(atom_array.res_name[token_starts]), dtype=torch.int64
    )
    features["restype"] = encode_one_hot(
        restype_index, len(STANDARD_RESIDUES_WITH_GAP_3)
    ).to(torch.int32)
    features["is_protein"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.PROTEIN,
        dtype=torch.int32,
    )
    features["is_rna"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.RNA,
        dtype=torch.int32,
    )
    features["is_dna"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.DNA,
        dtype=torch.int32,
    )
    features["is_ligand"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.LIGAND,
        dtype=torch.int32,
    )

    # Bonds
    features["token_bonds"] = create_token_bonds(
        atom_array, features["token_index"].numpy()
    )

    # Masks
    features["token_mask"] = create_token_mask(len(token_starts), token_budget)

    # Atomization
    features["num_atoms_per_token"] = torch.tensor(
        np.diff(token_starts_with_stop),
        dtype=torch.int32,
    )
    features["start_atom_index"] = torch.tensor(
        token_starts,
        dtype=torch.int32,
    )
    features["is_atomized"] = torch.tensor(
        atom_array.is_atomized[token_starts], dtype=torch.int32
    )

    # Ground-truth-specific features
    # TODO reorganize GT feature logic
    if is_gt:
        features["atom_positions"] = torch.nan_to_num(
            torch.tensor(atom_array.coord, dtype=torch.float32)
        )
        features["atom_resolved_mask"] = torch.clamp(
            torch.ceil(torch.tensor(atom_array.occupancy, dtype=torch.float32)),
            min=0.0,
            max=1.0,
        )

    # Pad and return
    return pad_token_dim(
        features, token_budget, token_dim_index_map=token_dim_index_map
    )


@log_runtime_memory(runtime_dict_key="runtime-target-structure-feat")
def featurize_target_gt_structure_af3(
    atom_array_cropped: AtomArray,
    atom_array_gt: AtomArray,
    token_budget: int,
) -> dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]:
    """Wraps featurize_structure_af3 for creating target AND gt structure features.

    Expects the cropped and duplicate-expanded AtomArray as input. The target structure
    features are a flat dictionary, while the ground truth features are nested in a
    subdictionary under the 'ground_truth' key.

    Args:
        atom_array_cropped (AtomArray):
            AtomArray of the target structure.
        atom_array_gt (AtomArray):
            AtomArray of the duplicate-expanded ground truth structure.
        token_budget (int):
            Crop size.

    Returns:
        dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]:
            Target and ground truth features. Ground truth features are nested
            in a subdictionary under the 'ground_truth' key.
    """
    token_dim_index_map = {
        "residue_index": [-1],
        "token_index": [-1],
        "asym_id": [-1],
        "entity_id": [-1],
        "sym_id": [-1],
        "restype": [-2],
        "is_protein": [-1],
        "is_rna": [-1],
        "is_dna": [-1],
        "is_ligand": [-1],
        "token_bonds": [-1, -2],
        "num_atoms_per_token": [-1],
        "is_atomized": [-1],
        "start_atom_index": [-1],
    }
    features_target = featurize_structure_af3(
        atom_array_cropped,
        token_budget,
        token_dim_index_map=token_dim_index_map,
        is_gt=False,
    )
    features_gt = featurize_structure_af3(
        atom_array_gt, token_budget, token_dim_index_map=token_dim_index_map, is_gt=True
    )
    features_target["ground_truth"] = features_gt
    return features_target
