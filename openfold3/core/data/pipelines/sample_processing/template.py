"""Sample processing pipelines for templates."""

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.primitives.caches.format import DatasetCache
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.template import (
    TemplateSliceCollection,
    align_template_to_query,
    sample_templates,
)
from openfold3.core.data.resources.residues import MoleculeType


@log_runtime_memory(runtime_dict_key="runtime-template-proc")
def process_template_structures_af3(
    atom_array: AtomArray,
    n_templates: int,
    take_top_k: bool,
    template_cache_directory: Path,
    dataset_cache: DatasetCache,
    pdb_id: str,
    template_structures_directory: Path | None,
    template_structure_array_directory: Path | None,
    template_file_format: str,
    ccd: CIFFile | None,
) -> TemplateSliceCollection:
    """Processes template structures for all chains of a given target structure.

    Note: Only looks for templates for chains that have at least one atom in the crop.

    Args:
        atom_array (AtomArray):
            The cropped (training) or full (inference) atom array.
        n_templates (int):
            The number of templates to sample for each chain. As per section 2.4 of the
            AF3 SI, during training at most n_templates are taken randomly from the list
            of available templates for each chain. During inference, the top (sorted by
            e-value) n_templates are taken.
        take_top_k (bool):
            Whether to take the top K templates (True) or sample randomly (False).
        template_cache_directory (Path):
            The directory where the template cache is stored.
        dataset_cache (dict):
            The dataset cache.
        pdb_id (str):
            The PDB ID of the target structure.
        template_structures_directory (Path | None):
            The directory where the template structures are stored.
        template_structure_array_directory (Path | None):
            The directory where the preparsed and preprocessed template structure
            arrays are stored.
        template_file_format (str):
            The format of the template files.
        ccd (CIFFile | None):
            The parsed CCD file.

    Returns:
        TemplateSliceCollection:
            The sliced template atomarrays for each chain in the crop.
    """
    # Get protein chain IDs from the cropped atom array
    protein_chain_ids = np.unique(
        atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN].chain_id
    )
    if len(protein_chain_ids) == 0:
        return TemplateSliceCollection(template_slices={})

    # Iterate over protein chains in the atom array
    template_slices = {}
    for chain_id in protein_chain_ids:
        # Sample templates and fetch their data from the cache
        sampled_template_data = sample_templates(
            dataset_cache,
            template_cache_directory,
            n_templates,
            take_top_k,
            pdb_id,
            chain_id,
        )

        # Map token positions to template atom arrays
        template_slices[chain_id] = align_template_to_query(
            sampled_template_data,
            template_structures_directory,
            template_structure_array_directory,
            template_file_format,
            ccd,
            atom_array[atom_array.chain_id == chain_id],
        )

    return TemplateSliceCollection(template_slices=template_slices)
