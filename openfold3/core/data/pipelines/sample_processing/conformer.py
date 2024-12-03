import contextlib
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from biotite.structure import AtomArray
from rdkit.Chem import Mol

from openfold3.core.data.io.structure.mol import read_single_annotated_sdf
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.component import component_iter
from openfold3.core.data.primitives.structure.conformer import (
    ConformerGenerationError,
    add_conformer_atom_mask,
    compute_conformer,
    get_allnan_conformer,
    multistrategy_compute_conformer,
    set_single_conformer,
)
from openfold3.core.data.primitives.structure.labels import uniquify_ids


class ProcessedReferenceMolecule(NamedTuple):
    """Processed reference molecule instance with the reference conformer.

    Attributes:
        mol_id (str):
            Identifier like CCD ID or custom ID labeling each unique molecule. Used in
            featurization to infer which conformers originate from the same molecule.
        mol (Mol):
            RDKit Mol object of the reference conformer instance with either a generated
            conformer, or if conformer generation is not possible, the fallback
            conformer. The mol object also contains the following internal atom-wise
            attributes parsed from the reference .sdf file:
                - "annot_atom_name":
                    Atom names
                - "annot_used_atom_mask":
                    Mask for atoms that are not NaN in the conformer.
        in_array_mask (np.ndarray[np.bool]):
            Mask for atoms that are within the current target's atom array.
    """

    mol_id: str
    mol: Mol
    in_array_mask: np.ndarray[bool]


def get_processed_reference_conformer(
    mol_id: Mol,
    mol: Mol,
    mol_atom_array: AtomArray,
    preferred_confgen_strategy: Literal["default", "random_init", "use_fallback"],
    set_fallback_to_nan: bool = False,
) -> ProcessedReferenceMolecule:
    """Creates a ProcessedReferenceMolecule instance.

    This function takes in a reference molecule and its corresponding AtomArray, sets
    the conformer to use during featurization (either a newly generated one or the
    stored fallback conformer if generation is not possible), and determines which atoms
    of the conformer are not NaN. The latter is relevant for the CCD-derived fallback
    conformers which may contain NaN values.

    Args:
        mol_id (str):
            Identifier like CCD ID or custom ID labeling each unique molecule.
        mol (Mol):
            RDKit Mol object of the reference conformer instance.
        mol_atom_array (AtomArray):
            AtomArray of the target conformer instance to determine which atoms of the
            reference conformer are present in the structure.
        preferred_confgen_strategy (str):
            Preferred strategy for conformer generation. If the strategy is
            "use_fallback" or the conformer generation fails, the fallback
            conformer is used.
        set_fallback_to_nan (bool, optional):
            If True, the fallback conformer is set to NaN. This is mostly relevant for
            the special case where the fallback conformer was derived from CCD model
            coordinates but the corresponding PDB ID is in the test set. Defaults to
            False.

    Returns:
        ProcessedReferenceMolecule:
            Processed reference molecule instance.
    """
    # Copy mol
    mol = Mol(mol)

    # Ensure mol has only one fallback conformer
    assert mol.GetNumConformers() == 1

    # Get atom names from RDKit mol and AtomArray
    mol_atom_names = [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]
    array_atom_names = mol_atom_array.atom_name

    # Necessary because multi-residue ligands like glycans may have repeated atom names
    # TODO: this is probably brittle, imagine a case where the spatial crop skips an
    # intermediate monomer and then what should be C1 C3 becomes C1 C2
    # FIX: run uniquification before cropping OR make use of original indices kept after
    # cropping
    mol_atom_names = np.array(uniquify_ids(mol_atom_names))
    array_atom_names = np.array(uniquify_ids(array_atom_names))

    # Set mask for atoms that are in the current target's atom array
    in_array_mask = np.isin(mol_atom_names, array_atom_names)

    # If we can't use the fallback conformer (e.g. if it was derived from a PDB ID in
    # the test set), we set it to NaN
    if set_fallback_to_nan:
        conf = get_allnan_conformer(mol)
        mol = set_single_conformer(mol, conf)

        # Adjust the non-NaN mask (to all-False)
        mol = add_conformer_atom_mask(mol)

    ## Overwrite the fallback conformer with a new conformer if possible
    if preferred_confgen_strategy != "use_fallback":
        # If the new conformer generation fails, the fallback conformer is used
        with contextlib.suppress(ConformerGenerationError):
            if preferred_confgen_strategy == "default":
                # Try with default, then use random init, then use fallback (technically
                # default should not fail because we already tried the strategy in
                # preprocessing)
                mol, conf_id, _ = multistrategy_compute_conformer(mol)
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

    return ProcessedReferenceMolecule(mol_id, mol, in_array_mask)


@log_runtime_memory(runtime_dict_key="runtime-ref-conf-proc")
def get_ref_conformer_data_af3(
    atom_array: AtomArray,
    per_chain_metadata: dict,
    reference_mol_metadata: dict,
    reference_mol_dir: Path,
) -> list[ProcessedReferenceMolecule]:
    """Extracts reference conformer data from AtomArray.

    Args:
        atom_array (AtomArray):
            Atom array of the whole crop.
        per_chain_metadata (dict):
            The "chains" subdictionary of the particular target's dataset cache entry.
        reference_mol_metadata (dict):
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
    for component_array in component_iter(atom_array, per_chain_metadata):
        chain_id = component_array.chain_id[0]

        # Either get the reference molecule ID from the chain metadata (in case of a
        # ligand chain) or use the residue name (in case of a single component of a
        # biopolymer)
        ref_mol_id = per_chain_metadata[chain_id].get(
            "reference_mol_id", component_array.res_name[0]
        )

        mol = read_single_annotated_sdf_cached(reference_mol_dir / f"{ref_mol_id}.sdf")

        processed_conformers.append(
            get_processed_reference_conformer(
                ref_mol_id,
                mol,
                component_array,
                reference_mol_metadata[ref_mol_id]["conformer_gen_strategy"],
            )
        )

    return processed_conformers
