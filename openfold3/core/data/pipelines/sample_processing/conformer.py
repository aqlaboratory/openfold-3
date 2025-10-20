import contextlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from biotite.structure import AtomArray
from func_timeout import FunctionTimedOut
from rdkit import Chem
from rdkit.Chem import Mol

from openfold3.core.data.io.structure.mol import read_single_annotated_sdf
from openfold3.core.data.primitives.caches.format import (
    DatasetChainData,
    DatasetReferenceMoleculeData,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.conformer import (
    ConformerGenerationError,
    add_conformer_atom_mask,
    compute_conformer,
    get_allnan_conformer,
    get_cropped_permutations,
    get_name_match_argsort,
    multistrategy_compute_conformer,
    set_single_conformer,
)
from openfold3.core.data.primitives.structure.labels import (
    AtomArrayView,
    component_view_iter,
    uniquify_ids,
)


@dataclass
class ProcessedReferenceMolecule:
    """Processed reference molecule instance with the reference conformer.

    Attributes:
        mol (Mol):
            RDKit Mol object of the reference conformer instance with either a generated
            conformer, or if conformer generation is not possible, the fallback
            conformer. The mol object also contains the following internal atom-wise
            attributes parsed from the reference .sdf file:
                - "annot_atom_name":
                    Atom names
                - "annot_used_atom_mask":
                    Mask for atoms that are not NaN in the conformer.
        in_crop_mask (np.ndarray[np.bool]):
            Mask for atoms that are within the current cropped crop's AtomArray.
        component_id (int):
            Original component ID in the cropped AtomArray.
        permutations (list[np.ndarray[np.int]]):
            List of symmetry-equivalent atom permutations for the reference conformer,
            adjusted to only map to in-crop atoms, and only draw from the pool of atoms
            present in the GT.
    """

    mol: Mol
    in_crop_mask: np.ndarray[bool]

    # TODO: make optional in inference
    component_id: int | None = None
    permutations: list[np.ndarray[int]] | None = None


@log_runtime_memory(runtime_dict_key="runtime-ref-conf-proc-fetch", multicall=True)
def get_processed_reference_conformer(
    mol: Mol,
    mol_atom_array: AtomArrayView | AtomArray,
    preferred_confgen_strategy: Literal["default", "random_init", "use_fallback"],
    set_fallback_to_nan: bool = False,
    max_permutations: int = 1_000,
) -> ProcessedReferenceMolecule:
    """Creates a ProcessedReferenceMolecule instance.

    This function takes in a reference molecule and its corresponding AtomArray, sets
    the conformer to use during featurization (either a newly generated one or the
    stored fallback conformer if generation is not possible), and determines which atoms
    of the conformer are not NaN. The latter is relevant for the CCD-derived fallback
    conformers which may contain NaN values.

    Args:
        mol (Mol):
            RDKit Mol object of the reference conformer instance.
        mol_atom_array (AtomArray | AtomArrayView):
            AtomArray or AtomArrayView of the target conformer instance to determine
            which atoms of the reference conformer are present in the structure.
        preferred_confgen_strategy (str):
            Preferred strategy for conformer generation. If the strategy is
            "use_fallback" or the conformer generation fails, the fallback
            conformer is used.
        set_fallback_to_nan (bool, optional):
            If True, the fallback conformer is set to NaN. This is mostly relevant for
            the special case where the fallback conformer was derived from CCD model
            coordinates but the corresponding PDB ID is in the test set. Defaults to
            False.
        max_permutations (int, optional):
            Maximum number of symmetry-equivalent atom permutations to generate for the
            reference conformer. Defaults to 1_000.

    Returns:
        ProcessedReferenceMolecule:
            Processed reference molecule instance.
    """
    # Copy mol
    mol = Mol(mol)

    # Extract component ID
    component_id = mol_atom_array.component_id[0]

    # Ensure mol has only one fallback conformer
    assert mol.GetNumConformers() == 1

    # Get uniquified atom names from RDKit mol and AtomArray (necessary because
    # multi-residue ligands can have duplicated atom names)
    conf_atom_names = np.array(
        uniquify_ids([atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()])
    )
    gt_atom_names = mol_atom_array.atom_name_unique

    # Reorder atoms if the name orders are different between the refence conformer and
    # mol atom array
    reorder_index = get_name_match_argsort(conf_atom_names, gt_atom_names)

    conf_atom_names = conf_atom_names[reorder_index]
    mol = Chem.RenumberAtoms(mol, reorder_index.tolist())

    # Ensure that all atoms in the ground-truth structure are also in the loaded
    # conformer data
    assert np.isin(gt_atom_names, conf_atom_names).all()

    # Mask for atoms that are in the ground-truth array
    in_gt_mask = np.isin(conf_atom_names, gt_atom_names)

    assert (conf_atom_names[in_gt_mask] == gt_atom_names).all()

    # Mask for atoms that are in the crop itself
    in_crop_atom_names = gt_atom_names[mol_atom_array.crop_mask]
    in_crop_mask = np.isin(conf_atom_names, in_crop_atom_names)

    cropped_permutations = get_cropped_permutations(
        mol=mol,
        in_gt_mask=in_gt_mask,
        in_crop_mask=in_crop_mask,
        max_permutations=max_permutations,
    )

    # Ensure that the cropped permutations map exactly to number of in-crop atoms
    assert cropped_permutations.shape[1] == len(in_crop_atom_names)

    # Ensure that the cropped pernutation indices are within the bounds of the
    # ground-truth
    assert cropped_permutations.max() <= len(gt_atom_names)

    # If we can't use the fallback conformer (e.g. if it was derived from a PDB ID in
    # the test set), we set it to NaN
    if set_fallback_to_nan:
        conf = get_allnan_conformer(mol)
        mol = set_single_conformer(mol, conf)

        # Adjust the non-NaN mask (to all-False)
        mol = add_conformer_atom_mask(mol)

    ## Overwrite the fallback conformer with a new conformer if possible
    if preferred_confgen_strategy != "use_fallback":
        # If the new conformer generation fails, the below code is skipped and the
        # fallback conformer is used
        with contextlib.suppress(ConformerGenerationError, FunctionTimedOut):
            if preferred_confgen_strategy == "default":
                # Try with default, then use random init, then use fallback (technically
                # default should not fail because we already tried the strategy in
                # preprocessing)
                mol, conf_id, _ = multistrategy_compute_conformer(
                    mol, remove_hs=True, timeout_standard=30.0, timeout_rand_init=30.0
                )
                conf = mol.GetConformer(conf_id)
            elif preferred_confgen_strategy == "random_init":
                # Try with random init, then use fallback (technically this also should
                # not fail). We do not use the default strategy here as a fallback
                # because this was already tried previously in preprocessing if
                # random_init was chosen.
                mol, conf_id = compute_conformer(mol, use_random_coord_init=True)
                conf = mol.GetConformer(conf_id)
            else:
                raise ValueError(
                    f"Conformer generation strategy '{preferred_confgen_strategy}' "
                    f"is not supported."
                )

            # Set the single conformer
            mol = set_single_conformer(mol, conf)

            # Adjust the non-NaN mask (will be all-True because conformer generation
            # worked)
            mol = add_conformer_atom_mask(mol)

    return ProcessedReferenceMolecule(
        mol=mol,
        component_id=component_id,
        in_crop_mask=in_crop_mask,
        permutations=cropped_permutations,
    )


# TODO: update the docstring here to make clearer how this is operating on the full atom
# array but only returning in-crop conformer data
@log_runtime_memory(runtime_dict_key="runtime-ref-conf-proc")
def get_reference_conformer_data_of3(
    atom_array: AtomArray,
    per_chain_metadata: DatasetChainData,
    reference_mol_metadata: DatasetReferenceMoleculeData,
    reference_mol_dir: Path,
) -> list[ProcessedReferenceMolecule]:
    """Extracts reference conformer data from AtomArray.

    Args:
        atom_array (AtomArray):
            Atom array of the whole crop.
        per_chain_metadata (DatasetChainData):
            The "chains" subdictionary of the particular target's dataset cache entry.
        reference_mol_metadata (DatasetReferenceMoleculeData):
            The "reference_molecule_data" subdictionary of the dataset cache.
        reference_mol_dir (Path):
            Path to the directory containing the reference molecule .sdf files generated
            in preprocessing.

    Returns:
        list[ProcessedReferenceConformer]:
            List of processed reference conformer instances.
    """
    # Cache the SDF parser to reduce file I/O (especially for frequently occurring
    # reference molecules like standard residues)
    read_single_annotated_sdf_cached = lru_cache(maxsize=100)(read_single_annotated_sdf)

    processed_conformers = []

    # Fill the list of processed reference conformers with all relevant information
    for component_array_view in component_view_iter(atom_array):
        # Skip if no atoms are in the crop
        if not component_array_view.crop_mask.any():
            continue

        chain_id = component_array_view.chain_id[0]

        # Either get the reference molecule ID from the chain metadata (in case of a
        # ligand chain) or use the residue name (in case of a single component of a
        # biopolymer)
        ref_mol_id = getattr(per_chain_metadata[chain_id], "reference_mol_id", None)
        if ref_mol_id is None:
            ref_mol_id = component_array_view.res_name[0]

        mol = read_single_annotated_sdf_cached(reference_mol_dir / f"{ref_mol_id}.sdf")

        processed_conformers.append(
            get_processed_reference_conformer(
                mol=mol,
                mol_atom_array=component_array_view,
                preferred_confgen_strategy=reference_mol_metadata[
                    ref_mol_id
                ].conformer_gen_strategy,
                set_fallback_to_nan=reference_mol_metadata[
                    ref_mol_id
                ].set_fallback_to_nan,
            )
        )

    return processed_conformers
