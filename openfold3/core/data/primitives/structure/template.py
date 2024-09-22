"""Primitives for processing templates structures."""

import dataclasses
import json
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.unresolved import (
    add_unresolved_polymer_residues,
)


@np.vectorize(excluded=["map_dict"])
def _map_idx_vectorized(k, map_dict):
    return map_dict.get(k, -1)


@dataclasses.dataclass(frozen=False)
class TemplateSlice:
    """Class storing cropped template structure and query-to-template map.

    Attributes:
        atom_array (AtomArray):
            AtomArray of the cropped template structure.
        idx_map (dict[int, int]):
            Dictionary mapping tokens that fall into the crop to corresponding residue
            indices in the matching alignment."""

    atom_array: AtomArray
    idx_map: dict[int, int]


@dataclasses.dataclass(frozen=False)
class TemplateSliceCollection:
    """Class for all cropped templates of all chains of a query assembly.

    Note: only contains templates for chains that fall into the crop. Lists
    for chains that have no templates are empty.

    Attributes:
        template_slices (dict[int, list[TemplateSlice]]):
            Dict mapping query chain ID to a list of TemplateSlice objects."""

    template_slices: dict[str, list[TemplateSlice]]


def get_query_structure_res_ids(
    atom_array_cropped: AtomArray, chain_id: int
) -> np.ndarray[int]:
    """_summary_

    Args:
        atom_array_cropped (AtomArray): _description_
        chain_id (int): _description_

    Returns:
        np.ndarray[int]: _description_
    """
    atom_array_cropped_chain = atom_array_cropped[
        atom_array_cropped.chain_id == chain_id
    ]
    cropped_query_res_starts = struc.get_residue_starts(atom_array_cropped_chain)
    return atom_array_cropped_chain[cropped_query_res_starts].res_id.astype(int)


def get_valid_templates(dataset_cache: dict, pdb_id: str, chain_id: int) -> list[str]:
    """_summary_

    Args:
        dataset_cache (dict): _description_
        pdb_id (str): _description_
        chain_id (int): _description_

    Returns:
        list[str]: _description_
    """

    return dataset_cache[pdb_id]["chains"][chain_id]["valid_templates"]


def sample_template_count(
    valid_templates: list[str], n_templates: int, is_train: bool
) -> int:
    """_summary_

    Args:
        valid_templates (list[str]): _description_
        n_templates (int): _description_
        is_train (bool): _description_

    Returns:
        int: _description_
    """
    if is_train:
        return np.min([np.random.randint(0, len(valid_templates)), n_templates])
    else:
        return n_templates


def parse_template_cache_entry(
    template_cache_path: Path, dataset_cache: dict, pdb_id: str, chain_id: str
) -> dict:
    with open(
        template_cache_path
        / Path(
            "{}.json".format(
                dataset_cache[pdb_id]["chains"][chain_id]["representative_id"]
            )
        )
    ) as f:
        template_cache = json.load(f)
    return template_cache


def slice_templates_for_chain(
    template_cache_entry: dict,
    k: int,
    template_structures_path: Path,
    ccd: CIFFile,
    cropped_query_res_ids: np.ndarray[int],
    valid_templates: list[str],
):
    cropped_templates = []

    # Randomly sample k templates
    sampled_templates = np.random.choice(valid_templates, k, replace=False)

    # iterate over the k templates
    for template_pdb_chain_id in sampled_templates:
        template_pdb_id, template_chain_id = template_pdb_chain_id.split("_")
        # parse their cif files into an atom array
        cif_file, atom_array_template_assembly = parse_mmcif(
            template_structures_path / Path(f"{template_pdb_id}.bcif")
        )
        # Add unresolved residues
        atom_array_template_assembly = add_unresolved_polymer_residues(
            atom_array_template_assembly, get_cif_block(cif_file), ccd
        )

        # Get matching chain from the template assembly
        atom_array_template_chain = atom_array_template_assembly[
            atom_array_template_assembly.label_asym_id == template_chain_id
        ]

        # get aligned residues using idx_map
        idx_map = {
            int(k): v
            for k, v in template_cache_entry[template_pdb_chain_id]["idx_map"].items()
        }
        cropped_template_res_ids = _map_idx_vectorized(
            cropped_query_res_ids, map_dict=idx_map
        )
        atom_array_cropped_template = atom_array_template_chain[
            np.isin(
                atom_array_template_chain.res_id.astype(int),
                cropped_template_res_ids,
            )
        ]

        # Add to list of cropped atom arrays for this chain
        cropped_templates.append(
            TemplateSlice(
                atom_array=atom_array_cropped_template,
                idx_map=idx_map,
            )
        )

    return cropped_templates
