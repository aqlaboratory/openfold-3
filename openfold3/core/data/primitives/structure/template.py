"""Primitives for processing templates structures."""

import dataclasses
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.caches.format import DatasetCache
from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.structure.cleanup import (
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_waters,
)
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.unresolved import (
    add_unresolved_atoms,
)


@dataclasses.dataclass(frozen=False)
class TemplateCacheEntry:
    """Class storing the template alignment and query-to-template map.

    Attributes:
        template_pdb_chain_id (str):
            The PDB+chain ID of the template structure.
        index (int):
            The row index of the template hit in the hmmsearch+hmmalign alignment.
        release_date (str):
            The release date of the template structure.
        idx_map (np.ndarray[int]):
            Dictionary mapping tokens that fall into the crop to corresponding residue
            indices in the matching alignment."""

    index: int
    release_date: str
    idx_map: np.ndarray[int]


@dataclasses.dataclass(frozen=False)
class TemplateSlice:
    """An AtomArray wrapper class for also storing the token positions.

    Attributes:
        atom_array (AtomArray):
            The AtomArray of the template. During training, this only contains the
            residues that align to query residues in the crop. During inference, this
            contains all residues of the template chain aligned to the query chain.
        query_token_positions (np.ndarray[int]):
            The token positions in the query structure.
        template_residue_repeats (np.ndarray[int]):
            Number of times to repeat each residue. Used for expanding template residue
            features for template residues that align to query residues tokenized per
            atom.
    """

    atom_array: AtomArray
    query_token_positions: np.ndarray[int]
    template_residue_repeats: np.ndarray[int]


@dataclasses.dataclass(frozen=False)
class TemplateSliceCollection:
    """Class for all cropped templates of all chains of a query assembly.

    Note: only contains templates for chains that have at least one residue that aligns
    to a query residue in the crop. Lists for chains that have no such templates are
    empty.

    Attributes:
        template_slices (dict[int, list[TemplateSlice]]):
            Dict mapping query chain ID to a list of cropped template AtomArray objects
            with the query token position to template residue ID map.
    """

    template_slices: dict[str, list[TemplateSlice]]


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


def sample_templates(
    dataset_cache: DatasetCache,
    template_cache_directory: Path,
    n_templates: int,
    take_top_k: bool,
    pdb_id: str,
    chain_id: str,
) -> dict[str, TemplateCacheEntry] | dict[None]:
    """Samples templates to featurize for a given chain.

    Follows the logic in section 2.4 of the AF3 SI.

    Args:
        dataset_cache (dict):
            The dataset cache.
        template_cache_directory (Path):
            The directory where the template cache is stored.
        n_templates (int):
            The max number of templates to sample for each chain.
        take_top_k (bool):
            Whether to take the top K templates (True) or sample randomly (False).
        pdb_id (str):
            The PDB ID of the query structure.
        chain_id (str):
            The chain ID for which to sample the templates.

    Returns:
        dict[str, TemplateCacheEntry] | dict[None]:
            The sampled template data per chain given chain.
    """
    chain_data = dataset_cache.structure_data[pdb_id].chains[chain_id]
    template_ids = chain_data.template_ids
    l = len(template_ids)
    if l == 0:
        return {}

    # Sample actual number of templates to use
    if take_top_k:
        k = np.min([l, n_templates])
    else:
        k = np.min([np.random.randint(0, l), n_templates])

    if k > 0:
        # Load template cache numpy file
        template_file_name = chain_data.alignment_representative_id + ".npz"
        template_cache = np.load(
            template_cache_directory / Path(template_file_name), allow_pickle=True
        )

        # Unpack into dict
        template_cache = {key: value.item() for key, value in template_cache.items()}

        # Randomly sample k templates or take top k templates
        if take_top_k:
            sampled_template_ids = template_ids[:k]
        else:
            sampled_template_ids = np.random.choice(template_ids, k, replace=False)

        # Wrap each subdict in a TemplateCacheEntry
        return {
            template_id: TemplateCacheEntry(
                index=template_cache[template_id]["index"],
                release_date=template_cache[template_id]["release_date"],
                idx_map=template_cache[template_id]["idx_map"],
            )
            for template_id in sampled_template_ids
        }

    else:
        return {}


def subset_template_index_map(
    template_cache_entry: TemplateCacheEntry, atom_array_query_chain: AtomArray
) -> None:
    """Subsets the idx map to template residues that align to the query crop.

    Args:
        template_cache_entry (TemplateCacheEntry):
            An entry from the template cache, containing an n-by-2 numpy array, 1st col:
            query residue index, 2nd col: template residue index, only containing
            positions that are non-gapped in the aligned template sequence.
        atom_array_query_chain (AtomArray):
            The query atom array for the current query chain. During training, this only
            contains the residues that are in the crop. During inference, this contains
            all residues of the query chain.
    """
    idx_map = template_cache_entry.idx_map
    idx_map = idx_map[idx_map[:, 0] != -1, :]

    # Subset idx map to template residues that are in the query chain
    res_in_query = np.unique(atom_array_query_chain.res_id.astype(int))
    idx_map_in_crop = idx_map[np.where(np.isin(idx_map[:, 0], res_in_query))[0]]

    # Update template cache entry with idx map in crop
    template_cache_entry.idx_map = idx_map_in_crop


def map_token_pos_to_template_residues(
    template_cache_entry: TemplateCacheEntry,
    atom_array_query_chain: AtomArray,
    atom_array_template_chain: AtomArray,
) -> TemplateSlice:
    """Creates index maps for the template residues that align to the query chain.

    Note: during training, also subsets the template atom array to only contain residues
    that align to query residues in the crop.

    Args:
        template_cache_entry (TemplateCacheEntry):
            An entry from the template cache, containing an n-by-2 numpy array, 1st col:
            query residue index, 2nd col: template residue index, only containing
            positions that are non-gapped in the aligned template sequence.
        atom_array_query_chain (AtomArray):
            The query atom array for the current query chain. During training, this only
            contains the residues that are in the crop. During inference, this contains
            all residues of the query chain.
        atom_array_template_chain (AtomArray):
            The template atom array for the current template chain.

    Returns:
        TemplateSlice:
            The atom array of a template containing only residues that align to query
            residues in the crop and the corresponding token positions and the mapping
            from query token positions to template residue IDs.
    """
    idx_map_in_crop = template_cache_entry.idx_map

    # Map query token positions to template residues
    query_token_atoms = atom_array_query_chain[get_token_starts(atom_array_query_chain)]

    # Get query tokens in the crop and to which template residues align
    query_token_atoms_aligned_cropped = query_token_atoms[
        np.isin(query_token_atoms.res_id, idx_map_in_crop[:, 0])
    ]
    # Expand residues tokenized per atom
    _, repeats = np.unique(query_token_atoms_aligned_cropped.res_id, return_counts=True)

    # Get template atom array with residues aligning to query residues in the crop
    atom_array_cropped_template = atom_array_template_chain[
        np.isin(
            atom_array_template_chain.res_id.astype(int),
            idx_map_in_crop[:, 1],
        )
    ]

    # Add token position annotation to template atom array mapping to the crop
    template_slice = TemplateSlice(
        atom_array=atom_array_cropped_template,
        query_token_positions=query_token_atoms_aligned_cropped.token_position,
        template_residue_repeats=repeats,
    )

    return template_slice


def align_template_to_query(
    sampled_template_data: dict[str, TemplateCacheEntry] | dict[None],
    template_structures_directory: Path,
    template_file_format: str,
    ccd: CIFFile,
    atom_array_query_chain: AtomArray,
) -> list[AtomArray]:
    """Identifies the subset of atoms in the template that align to the query crop.

    Args:
        sampled_template_data (dict[str, TemplateCacheEntry] | dict[None]):
            The sampled template data per chain given chain.
        template_structures_directory (Path):
            The directory where the template structures are stored.
        template_file_format (str):
            The format of the template structures.
        ccd (CIFFile):
            Parsed CCD file.
        atom_array_query_chain (AtomArray):
            The cropped atom array containing atoms of the current protein chain.).

    Returns:
        list[AtomArray]:
            List of template AtomArrays subset to residues that align to any residue in
            the query atom array and with added token ids in the query structure.
    """
    if len(sampled_template_data) == 0:
        return []

    template_slices = []
    # Iterate over the k templates
    for template_pdb_chain_id, template_cache_entry in sampled_template_data.items():
        # Subset the idx map to template residues that align to the query crop for
        # training and skip if the template is outside the crop
        subset_template_index_map(template_cache_entry, atom_array_query_chain)
        if template_cache_entry.idx_map.shape[0] == 0:
            continue

        # Parse template IDs
        template_pdb_id, template_chain_id = template_pdb_chain_id.split("_")

        # Parse the full template assembly
        cif_file, atom_array_template_assembly = parse_mmcif(
            template_structures_directory
            / Path(f"{template_pdb_id}.{template_file_format}")
        )
        # Clean up template atom array
        atom_array_template_assembly = remove_waters(atom_array_template_assembly)
        atom_array_template_assembly = remove_hydrogens(atom_array_template_assembly)
        atom_array_template_assembly = remove_non_CCD_atoms(
            atom_array_template_assembly, ccd
        )
        atom_array_template_assembly = add_unresolved_atoms(
            atom_array_template_assembly, get_cif_block(cif_file), ccd
        )

        # Get matching chain from the template assembly using the PDB assigned chain ID
        atom_array_template_chain = atom_array_template_assembly[
            atom_array_template_assembly.label_asym_id == template_chain_id
        ]

        # Create query token position to template residue ID map
        template_slice = map_token_pos_to_template_residues(
            template_cache_entry,
            atom_array_query_chain,
            atom_array_template_chain,
        )

        # Add to list of cropped + aligned template atom arrays for this chain
        # if might not be needed here?
        if len(template_slice.atom_array) > 0:
            template_slices.append(template_slice)

    return template_slices
