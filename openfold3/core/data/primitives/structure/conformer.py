import logging
from typing import Literal

import numpy as np
from func_timeout import FunctionTimedOut, func_set_timeout
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Mol

from openfold3.core.data.primitives.structure.component import (
    AnnotatedMol,
    safe_remove_all_hs,
    set_atomwise_annotation,
)

logger = logging.getLogger(__name__)


class ConformerGenerationError(ValueError):
    """An error that is raised when the generation of a conformer fails."""

    pass


@func_set_timeout(30)
def compute_conformer(
    mol: Mol, use_random_coord_init: bool = False, remove_hs: bool = True
) -> tuple[Mol, int]:
    """Computes a conformer with the ETKDGv3 strategy.

    Wrapper around RDKit's EmbedMolecule, using ETKDGv3, handling hydrogen addition and
    removal, and raising an explicit ConformerGenerationError instead of returning -1.
    A `FunctionTimedOut` exception is raised if conformer generation exceeds 30 seconds.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
        use_random_coord_init:
            Whether to initialize the conformer generation with random coordinates
            (recommended for failure cases or large molecules)
        remove_hs:
            Whether to remove hydrogens from the molecule after conformer generation.
            The function automatically adds hydrogens before conformer generation.

    Returns:
        mol:
            The molecule for which the 3D coordinates should be computed.
        conformer ID:
            The ID of the conformer that was generated.

    Raises:
        ConformerGenerationError:
            If the conformer generation fails.
    """
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        logger.warning(f"Failed to add hydrogens before conformer generation: {e}")

    strategy = AllChem.ETKDGv3()

    if use_random_coord_init:
        strategy.useRandomCoords = True

    strategy.clearConfs = False

    # Disable overly verbous conformer generation warnings
    blocker = rdBase.BlockLogs()
    conf_id = AllChem.EmbedMolecule(mol, strategy)
    del blocker

    if remove_hs:
        mol = safe_remove_all_hs(mol)

    if conf_id == -1:
        raise ConformerGenerationError("Failed to generate 3D coordinates")

    return mol, conf_id


# TODO: could improve warning handling of this to send less UFFTYPER warnings
def multistrategy_compute_conformer(
    mol: Mol,
    remove_hs: bool = True,
) -> tuple[Mol, int, Literal["default", "random_init"]]:
    """Computes 3D coordinates for a molecule trying different initializations.

    Tries to compute 3D coordinates for a molecule using the standard RDKit ETKDGv3
    strategy. If this fails or times out, it falls back to using a different
    initialization for ETKDGv3 with random starting coordinates. If this also fails
    or times out, a `ConformerGenerationError` is raised.

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
    # Try standard ETKDGv3 strategy first
    try:
        mol, conf_id = compute_conformer(
            mol, use_random_coord_init=False, remove_hs=remove_hs
        )
    except (ConformerGenerationError, FunctionTimedOut) as e:
        logger.warning(
            f"Exception when trying standard conformer generation: {e}, "
            + "trying random initialization"
        )

        # Try random coordinates as fallback
        try:
            mol, conf_id = compute_conformer(
                mol, use_random_coord_init=True, remove_hs=remove_hs
            )
        except (ConformerGenerationError, FunctionTimedOut) as e:
            logger.warning(
                "Exception when trying conformer generation with random "
                + f"initialization: {e}"
            )
            raise ConformerGenerationError("Failed to generate 3D coordinates") from e
        else:
            success_strategy = "random_init"
    else:
        success_strategy = "default"

    return mol, conf_id, success_strategy


def add_conformer_atom_mask(mol: Mol) -> AnnotatedMol:
    """Adds a mask of valid atoms, masking out NaN conformer coordinates.

    This uses the first conformer in the molecule to find atoms with NaN coordinates and
    storing them in an appropriate mask attribute. NaN coordinates are usually an
    artifact of the CCD data, which can have missing coordinates for the stored ideal or
    model coordinates.

    Args:
        mol:
            The molecule for which the mask should be added.

    Returns:
        Mol with the mask added as an atom-wise property under the key
        "used_atom_mask_annot".
    """
    conf = mol.GetConformer()
    all_coords = conf.GetPositions()

    mask = (~np.any(np.isnan(all_coords), axis=1)).tolist()

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
        conf.SetAtomPosition(atom_id, (np.nan, np.nan, np.nan))

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
) -> tuple[AnnotatedMol, Literal["default", "random_init", "use_fallback"]]:
    """Retains a single "fallback conformer" in the molecule.

    The purpose of this function is two-fold: The first is to set a single set of
    coordinates for the molecule that should be used as a fallback in case the
    on-the-fly conformer generation fails. The second purpose is to already "test out"
    conformer generation strategies on the fallback conformer and store the strategy
    that worked, so that the featurization pipeline can use the same strategy to
    generate new conformers during training.

    To set the fallback conformer, this function uses the following strategy:
        1. Try to generate a conformer with `compute_conformer`, tracking the returned
           conformer-generation strategy. If successful, set this computed conformer as
           the fallback conformer. Note that this computed conformer will almost never
           be used, as the featurization pipeline will be able to generate a new
           conformer on-the-fly if the conformer generation already worked here.
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
            The strategy that should be used for conformer generation for this molecule
            during featurization:
                - "default": The standard ETKDGv3 strategy
                - "random_init": The ETKDGv3 strategy with random initialization
                - "use_fallback": Conformer generation is not possible and the stored
                  fallback conformer should be used.
    """
    # Test if conformer generation is possible
    try:
        mol, conf_id, strategy = multistrategy_compute_conformer(mol)
        conf = mol.GetConformer(conf_id)
    except ConformerGenerationError:
        strategy = "use_fallback"
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
