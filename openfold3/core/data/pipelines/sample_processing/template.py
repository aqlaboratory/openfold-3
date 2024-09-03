"""Sample processing pipelines for templates."""

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.primitives.structure.template import (
    TemplateSliceCollection,
    get_query_structure_res_ids,
    parse_template_cache_entry,
    sample_template_count,
    slice_templates_for_chain,
)
from openfold3.core.data.resources.residues import MoleculeType


def process_template_structures_af3(
    atom_array_cropped: AtomArray,
    n_templates: int,
    is_train: bool,
    template_cache_path: Path,
    dataset_cache: dict,
    pdb_id: str,
    template_structures_path: Path,
    ccd: CIFFile,
) -> TemplateSliceCollection:
    # Get protein chain IDs
    protein_chain_ids = np.unique(
        atom_array_cropped[
            atom_array_cropped.molecule_type_id == MoleculeType.PROTEIN
        ].chain_id
    )

    # Iterate over chains in the crop
    template_slice_collection = TemplateSliceCollection(template_slices={})
    for chain_id in protein_chain_ids:
        # Get residue ids for the current chain
        cropped_query_res_ids = get_query_structure_res_ids(
            atom_array_cropped, chain_id
        )   

        # Get list of templates for chain
        valid_templates = dataset_cache[pdb_id]["chains"][chain_id]["valid_templates"]

        # Get actual number of templates to sample for this chain
        k = sample_template_count(valid_templates, n_templates, is_train)

        if k > 0:
            # Get template cache
            template_cache_entry = parse_template_cache_entry(
                template_cache_path, dataset_cache, pdb_id, chain_id
            )

            # Slice templates for the current chain
            cropped_templates = slice_templates_for_chain(
                template_cache_entry,
                k,
                template_structures_path,
                ccd,
                cropped_query_res_ids,
                valid_templates,
            )

            # Add the sliced templates to the template slice collection

        template_slice_collection.template_slices[chain_id] = cropped_templates

    return template_slice_collection
