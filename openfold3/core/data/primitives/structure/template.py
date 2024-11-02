"""Primitives for processing templates structures."""

import dataclasses
import json
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.unresolved import (
    add_unresolved_polymer_residues,
)


@dataclasses.dataclass(frozen=False)
class TemplateCacheEntry:
    """Class storing the template alignment and query-to-template map.

    Attributes:
        template_pdb_chain_id (str):
            The PDB+chain ID of the template structure.
        e_value (int):
            The e-value of the template structure.
        release_date (str):
            The release date of the template structure.
        idx_map (dict[str, int]):
            Dictionary mapping tokens that fall into the crop to corresponding residue
            indices in the matching alignment."""

    template_pdb_chain_id: str
    e_value: int
    release_date: str
    idx_map: dict[str, int]


@dataclasses.dataclass(frozen=False)
class TemplateSlice:
    """Class storing cropped template structure and query-to-template map.

    Note: a TemplateSlice only contains residues of the template that
        1. are aligned to the query structure in the hhsearch alignment
        2. fall into the crop of then query structure.

    Attributes:
        atom_array (AtomArray):
            AtomArray of the cropped template structure.
        res_id_res_id_map (dict[int, int]):
            Dictionary mapping query residue IDs to template residue IDs
            determined by the alignment.
        res_id_token_position_map (dict[int, int]):
            Dictionary mapping template residue IDs to token positions.
    """

    atom_array: AtomArray
    res_id_res_id_map: dict[int, int]
    res_id_token_position_map: dict[int, int]


@dataclasses.dataclass(frozen=False)
class TemplateSliceCollection:
    """Class for all cropped templates of all chains of a query assembly.

    Note: only contains templates for chains that fall into the crop. Lists
    for chains that have no templates are empty.

    Attributes:
        template_slices (dict[int, list[AtomArray]]):
            Dict mapping query chain ID to a list of cropped template AtomArray objects.
    """

    template_slices: dict[str, list[AtomArray]]


def get_query_structure_res_ids(atom_array_cropped_chain: AtomArray) -> np.ndarray[int]:
    """Retrieves residue IDs of the query structure for a given chain.

    Args:
        atom_array_cropped_chain (AtomArray):
            The cropped atom array for all chains.

    Returns:
        np.ndarray[int]:
            Residue IDs of the query structure for the given chain.
    """
    cropped_query_res_starts = struc.get_residue_starts(atom_array_cropped_chain)
    return atom_array_cropped_chain[cropped_query_res_starts].res_id.astype(int)


def fetch_template_ids(dataset_cache: dict, pdb_id: str, chain_id: int) -> list[str]:
    """Parses the template IDs for a given chain from the dataset cache.

    Args:
        dataset_cache (dict):
            The dataset cache.
        pdb_id (str):
            The PDB ID of the query structure.
        chain_id (int):
            The chain ID for which to retrieve the template IDs.

    Returns:
        list[str]:
            List of template PDB+chain IDs for the given chain.
    """

    return dataset_cache["structure_data"][pdb_id]["chains"][chain_id]["template_ids"]


def sample_template_count(
    template_pdb_chain_ids: list[str], n_templates: int, is_train: bool
) -> int:
    """Samples the actual number of templates to use for a given chain.

    Follows the logic in section 2.4 of the AF3 SI.

    Args:
        template_pdb_chain_ids (list[str]):
            List of valid template PDB+chain IDs.
        n_templates (int):
            The max number of templates to sample for each chain.
        is_train (bool):
            Whether the current processing is for training or not.

    Returns:
        int:
            The actual number of templates to sample for this chain.
    """
    if len(template_pdb_chain_ids) == 0:
        return 0

    if is_train:
        return np.min([np.random.randint(0, len(template_pdb_chain_ids)), n_templates])
    else:
        return np.min(len(template_pdb_chain_ids), n_templates)


def parse_template_cache_entries(
    template_cache_directory: Path, dataset_cache: dict, pdb_id: str, chain_id: str
) -> dict[str, TemplateCacheEntry]:
    """Parses the template cache for a given chain.

    Args:
        template_cache_directory (Path):
            The directory where the template cache is stored.
        dataset_cache (dict):
            The dataset cache.
        pdb_id (str):
            The PDB ID of the query structure.
        chain_id (str):
            The chain ID for which to retrieve the template cache.

    Returns:
        dict: _description_
    """
    with open(
        template_cache_directory
        / Path(
            "{}.json".format(
                dataset_cache["structure_data"][pdb_id]["chains"][chain_id][
                    "alignment_representative_id"
                ]
            )
        )
    ) as f:
        template_cache = json.load(f)
    return {
        template_pdb_chain_id: TemplateCacheEntry(
            template_pdb_chain_id=template_pdb_chain_id,
            e_value=template_data["e_value"],
            release_date=template_data["release_date"],
            idx_map=template_data["idx_map"],
        )
        for template_pdb_chain_id, template_data in template_cache.items()
    }


@np.vectorize(excluded=["map_dict"])
def expand_alignment_to_cropped_query(k, map_dict):
    "Expands the residue index map to the cropped query sequence."
    return map_dict.get(k, -1)


def create_index_maps(
    idx_map: dict[int, int], atom_array_cropped_chain: AtomArray
) -> tuple[dict[int, int], dict[int, int]]:
    """Creates index maps for the template residues that align to the query crop.

    Args:
        idx_map (dict[int, int]):
            Dict mapping all query residue indices to template residue indices
            according to the hhsearch alignment.
        atom_array_cropped_chain (AtomArray):
            The cropped atom array for the current chain.

    Returns:
        tuple[dict[int, int], dict[int, int]]:
            Tuple containing:
            - Dictionary mapping query residue IDs that are in the crop to template
            residue IDs.
            - Dictionary mapping template residue IDs to token
    """
    # Subset idx map to template residues that are in the cropped query chain
    res_id_res_id_map = {
        k: v
        for k, v in idx_map.items()
        if k in atom_array_cropped_chain.res_id.astype(int)
    }

    # Create a map from template residue index to token position
    token_starts = get_token_starts(atom_array_cropped_chain)
    token_atoms = atom_array_cropped_chain[token_starts]
    res_id_token_position_map = {
        res_id_res_id_map[query_res_id]: token_position
        for query_res_id, token_position in zip(
            token_atoms.res_id, token_atoms.token_position
        )
        if query_res_id in res_id_res_id_map
    }

    return res_id_res_id_map, res_id_token_position_map


def slice_templates_for_chain(
    template_cache: dict,
    k: int,
    template_structures_directory: Path,
    template_file_format: str,
    ccd: CIFFile,
    atom_array_cropped_chain: AtomArray,
    template_pdb_chain_ids: list[str],
    is_train: bool,
) -> list[AtomArray]:
    """Identifies the subset of atoms in the template that align to the query crop.

    Args:
        template_cache (dict):
            The template cache for the current chain.
        k (int):
            The number of templates to use.
        template_structures_directory (Path):
            The directory where the template structures are stored.
        template_file_format (str):
            The format of the template structures.
        ccd (CIFFile):
            Parsed CCD file.
        atom_array_cropped_chain (AtomArray):
            The cropped atom array containing atoms of the current protein chain.
        template_pdb_chain_ids (list[str]):
            List of valid template PDB+chain IDs.
        is_train (bool):
            Whether the current processing is for training or not.

    Returns:
        list[AtomArray]:
            List of cropped template AtomArray objects for the current chain.
    """
    cropped_templates = []

    # Fetch residue ids and token starts for the current query chain
    cropped_query_res_ids = get_query_structure_res_ids(atom_array_cropped_chain)
    # Randomly sample k templates or take top k templates
    if is_train:
        sampled_templates = np.random.choice(template_pdb_chain_ids, k, replace=False)
    else:
        sampled_templates = template_pdb_chain_ids[:k]

    # Iterate over the k templates
    for template_pdb_chain_id in sampled_templates:
        # Parse IDs
        template_pdb_id, template_chain_id = template_pdb_chain_id.split("_")
        # Parse cif file into an atom array
        cif_file, atom_array_template_assembly = parse_mmcif(
            template_structures_directory
            / Path(f"{template_pdb_id}.{template_file_format}")
        )
        # Add unresolved residues
        atom_array_template_assembly = add_unresolved_polymer_residues(
            atom_array_template_assembly, get_cif_block(cif_file), ccd
        )

        # Get matching chain from the template assembly using the PDB assigned chain ID
        atom_array_template_chain = atom_array_template_assembly[
            atom_array_template_assembly.label_asym_id == template_chain_id
        ]

        # Fetch the residue index map
        idx_map = {
            int(k): v for k, v in template_cache[template_pdb_chain_id].idx_map.items()
        }

        # Subset idx map and create template residue idx to token position map
        res_id_res_id_map, res_id_token_position_map = create_index_maps(
            idx_map, atom_array_cropped_chain
        )

        # Get the set of residues in the template that align to any query residue in the
        # crop
        cropped_template_res_ids = np.vectorize(res_id_res_id_map.get)(
            cropped_query_res_ids
        )
        cropped_template_res_ids = cropped_template_res_ids[
            np.vectorize(lambda x: x is not None, otypes=[bool])(
                cropped_template_res_ids
            )
        ].astype(int)

        # Subset the template atom array
        atom_array_cropped_template = atom_array_template_chain[
            np.isin(
                atom_array_template_chain.res_id.astype(int),
                cropped_template_res_ids,
            )
        ]

        # Add query residue ids and res_id_token_position_map to the template atom array
        token_positions = np.vectorize(res_id_token_position_map.get)(
            cropped_template_res_ids
        )
        atom_array_cropped_template.set_annotation(
            "token_positions",
            struc.spread_residue_wise(atom_array_cropped_template, token_positions),
        )

        # Add to list of cropped template atom arrays for this chain
        cropped_templates.append(atom_array_cropped_template)

    return cropped_templates
