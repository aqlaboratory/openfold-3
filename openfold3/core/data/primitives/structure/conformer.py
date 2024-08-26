import logging
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Geometry import Point3D

from openfold3.core.data.primitives.structure.component import (
    AnnotatedMol,
    safe_remove_all_hs,
    set_atomwise_annotation,
)

logger = logging.getLogger(__name__)


class ConformerGenerationError(ValueError):
    """An error that is raised when the generation of a conformer fails."""

    pass


# TODO: could improve warning handling of this to send less UFFTYPER warnings
def compute_conformer(
    mol: Mol,
    remove_hs: bool = True,
) -> tuple[Mol, int, Literal["default", "random_init"]]:
    """Computes 3D coordinates for a molecule using RDKit.

    Tries to compute 3D coordinates for a molecule using the standard ETKDGv3 strategy.
    If this fails, it falls back to using a different initializion for ETKDGv3 with
    random starting coordinates. If this also fails, a `ConformerGenerationError` is
    raised.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
        remove_hs:
            Whether to remove hydrogens from the molecule after conformer generation.

    Returns:
        mol:
            The molecule for which the 3D coordinates should be computed.
        conformer ID:
            The ID of the conformer that was generated.
        strategy:
            The strategy that was used for conformer generation. Either "default" or
            "random_init".
    """
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        logger.warning(f"Failed to add hydrogens before conformer generation: {e}")

    # Try standard ETKDGv3 strategy first
    strategy = AllChem.ETKDGv3()
    strategy.ClearConfs = False
    try:
        conf_id = AllChem.EmbedMolecule(mol, strategy)
        if conf_id == -1:
            raise ConformerGenerationError("Failed to generate 3D coordinates")
    except Exception as e:
        logger.warning(f"Exception when calling EmbedMolecule: {e}, trying random init")

        # Try random coordinates as fallback
        try:
            strategy.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, strategy)
            if conf_id == -1:
                raise ConformerGenerationError("Failed to generate 3D coordinates")
        except Exception as e:
            logger.warning(
                f"Exception when calling EmbedMolecule with random coordinates: {e}"
            )
            mol = safe_remove_all_hs(mol)

            raise ConformerGenerationError("Failed to generate 3D coordinates") from e
        else:
            success_strategy = "random_init"
    else:
        success_strategy = "default"

    conf = mol.GetConformer(conf_id)
    conf.SetProp("name", "Computed")  # following pdbeccdutils ConformerType

    if remove_hs:
        mol = safe_remove_all_hs(mol)

    return mol, conf_id, success_strategy


def add_conformer_atom_mask(mol: Mol) -> AnnotatedMol:
    """Adds a mask of valid atoms, masking out NaN conformer coordinates.

    Uses the first conformer in the molecule.

    Args:
        mol:
            The molecule for which the mask should be added.

    Returns:
        Mol with the mask added as an atom-wise property under the key
        "used_atom_mask_annot".
    """
    conf = mol.GetConformer()
    all_coords = conf.GetPositions()

    mask = np.any(np.isnan(all_coords), axis=1)

    mol = set_atomwise_annotation(mol, "used_atom_mask", mask)

    return mol


def set_single_conformer(mol: Mol, conf: Chem.Conformer) -> Mol:
    """Replaces all stored conformers in a molecule with a single conformer."""
    mol = Chem.Mol(mol)  # make a copy, see rdkit issue #3817
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    return mol


def get_allnan_conformer(mol: Mol) -> Chem.Conformer:
    """Returns a conformer with all atoms set to NaN.

    Args:
        mol:
            The molecule for which the conformer should be generated.

    Returns:
        An RDKit conformer object with all coordinates set to NaN.
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_id in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_id, Point3D(0, 0, 0))

    return conf


def replace_nan_coords_with_zeros(mol: Mol) -> None:
    """Replaces all NaN coordinates in a molecule with zeros in-place.

    Args:
        mol:
            The molecule for which the NaN coordinates should be replaced.
    """
    for conf in mol.GetConformers():
        for atom_id in range(conf.GetNumAtoms()):
            if any(np.isnan(coord) for coord in conf.GetAtomPosition(atom_id)):
                conf.SetAtomPosition(atom_id, (0, 0, 0))


def resolve_and_format_fallback_conformer(
    mol: Mol,
) -> tuple[AnnotatedMol, Literal["default", "random_init", "failed"]]:
    """Retains a single "fallback conformer" in the molecule.

    The fallback conformer can be used by the data module if the conformer generation
    fails.

    To set the fallback conformer, this function uses the following strategy:
        1. Try to generate a conformer with `compute_conformer`, tracking the returned
           conformer-generation strategy. If successful, set this computed conformer as
           the fallback conformer.
        2. If this fails, try to use the first stored conformer. For CCD molecules
           created by `mol_from_pdbeccdutils_component`, this will correspond to the
           "Ideal" CCD conformer, or if not present, the "Model" conformer, following
           2.8 of the AlphaFold3 SI.
        3. If no stored conformer is available, set all coordinates to NaN.

    Args:
        mol:
            The molecule for which the fallback conformer should be resolved.

    Returns:
        mol:
            The molecule with a single fallback conformer set. The molecule object will
            have an additional atom-wise property "annot_used_atom_mask" which is set to
            "True" for all atoms with valid coordinates, and "False" for all atoms with
            NaN coordinates. The NaN coordinates themselves are set to 0, as .sdf files
            can't handle NaNs.
        strategy:
            The strategy that was used to generate the fallback conformer. Either
            "default" (ETKDGv3) or "random_init" (ETKDGv3 with random initialization).
            If conformer generation failed, this will be "failed".
    """
    # Test if conformer generation is possible
    try:
        mol, conf_id, strategy = compute_conformer(mol)
        conf = mol.GetConformer(conf_id)
    except ConformerGenerationError:
        strategy = "failed"
        # Try to use first stored conformer
        try:
            conf = next(mol.GetConformers())
        # If no stored conformer, use all-NaN conformer
        except StopIteration:
            conf = get_allnan_conformer(mol)

    # Remove all other conformers
    mol = set_single_conformer(mol, conf)

    # Add atom-wise mask of valid atoms in "annot_used_atom_mask" property
    mol = add_conformer_atom_mask(mol)

    # Set NaN coordinates to 0 (because .sdf can't handle NaNs)
    replace_nan_coords_with_zeros(mol)

    return mol, strategy
