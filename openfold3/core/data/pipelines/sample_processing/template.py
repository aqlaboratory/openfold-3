"""Sample processing pipelines for templates."""

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.primitives.structure.template import (
    TemplateSliceCollection,
    fetch_template_ids,
    parse_template_cache_entries,
    sample_template_count,
    slice_templates_for_chain,
)
from openfold3.core.data.resources.residues import MoleculeType


def process_template_structures_af3(
    atom_array_cropped: AtomArray,
    n_templates: int,
    is_train: bool,
    template_cache_directory: Path,
    dataset_cache: dict,
    pdb_id: str,
    template_structures_directory: Path,
    ccd: CIFFile,
) -> TemplateSliceCollection:
    """Processes template structures for all chains of a given target structure.

    Note: Only looks for templates for chains that have at least one atom in the crop.

    Args:
        atom_array_cropped (AtomArray):
            The cropped atom array.
        n_templates (int):
            The number of templates to sample for each chain. As per section 2.4 of the
            AF3 SI, during training at most n_templates are taken randomly from the list
            of available templates for each chain. During inference, the top (sorted by
            e-value) n_templates are taken.
        is_train (bool):
            Whether the current processing is for training or not.
        template_cache_directory (Path):
            The directory where the template cache is stored.
        dataset_cache (dict):
            The dataset cache.
        pdb_id (str):
            The PDB ID of the target structure.
        template_structures_directory (Path):
            The directory where the template structures are stored.
        ccd (CIFFile):
            The parsed CCD file.

    Returns:
        TemplateSliceCollection:
            The sliced templates for each chain in the crop.
    """
    # Get protein chain IDs from the cropped atom array
    protein_chain_ids = np.unique(
        atom_array_cropped[
            atom_array_cropped.molecule_type_id == MoleculeType.PROTEIN
        ].chain_id
    )

    # Iterate over protein chains in the crop
    template_slice_collection = TemplateSliceCollection(template_slices={})
    for chain_id in protein_chain_ids:
        # Subset the atom array to the current chain
        atom_array_cropped_chain = atom_array_cropped[
            atom_array_cropped.chain_id == chain_id
        ]

        # Get the preprocessed list of filtered templates
        template_pdb_chain_ids = fetch_template_ids(dataset_cache, pdb_id, chain_id)

        # Get actual number of templates to sample for this chain
        k = sample_template_count(template_pdb_chain_ids, n_templates, is_train)

        # Get the template slices (cropped atom arrays and residue maps) for the
        # current chain
        if k > 0:
            # Get template cache
            template_cache = parse_template_cache_entries(
                template_cache_directory, dataset_cache, pdb_id, chain_id
            )

            # Slice templates for the current chain
            cropped_templates = slice_templates_for_chain(
                template_cache,
                k,
                template_structures_directory,
                ccd,
                atom_array_cropped_chain,
                template_pdb_chain_ids,
                is_train,
            )

        else:
            cropped_templates = []

        # Add the sliced templates to the template slice collection
        template_slice_collection.template_slices[chain_id] = cropped_templates

    return template_slice_collection
