import logging
import random
from collections.abc import Iterable
from typing import Literal

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
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


def compute_conformer(
    mol: Mol,
    use_random_coord_init: bool = False,
    remove_hs: bool = True,
    timeout: float | None = 30.0,
) -> tuple[Mol, int]:
    """Computes a conformer with the ETKDGv3 strategy.

    Wrapper around RDKit's EmbedMolecule, using ETKDGv3, handling hydrogen addition and
    removal, and raising an explicit ConformerGenerationError instead of returning -1.
    A FunctionTimedOut exception is raised if conformer generation exceeds the
    given timeout.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
        use_random_coord_init:
            Whether to initialize the conformer generation with random coordinates
            (recommended for failure cases or large molecules)
        remove_hs:
            Whether to remove hydrogens from the molecule after conformer generation.
            The function automatically adds hydrogens before conformer generation.
        timeout:
            The maximum time in seconds to allow for conformer generation.
            Default value is 30 seconds. If None, no timeout is set.

    Returns:
        mol:
            The molecule for which the 3D coordinates should be computed.
        conformer ID:
            The ID of the conformer that was generated.

    Raises:
        ConformerGenerationError:
            If the conformer generation fails.

        FunctionTimedOut:
            If the conformer generation exceeds the given timeout.
    """
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        logger.warning(f"Failed to add hydrogens before conformer generation: {e}")

    strategy = AllChem.ETKDGv3()

    if use_random_coord_init:
        strategy.useRandomCoords = True

    strategy.clearConfs = False
    # RDKit always seems to start from some internal seed instead of a truly random seed
    # initialization if no seed is given, so we set a random seed here
    strategy.randomSeed = random.randint(0, 10**9)

    # Disable overly verbose conformer generation warnings
    blocker = rdBase.BlockLogs()

    if timeout is not None:
        conf_id = func_timeout(
            timeout=timeout, func=AllChem.EmbedMolecule, args=(mol, strategy)
        )
    else:
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
    timeout_standard: float | None = None,
    timeout_rand_init: float | None = None,
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
        timeout_standard:
            The maximum time in seconds to allow for conformer generation with the
            standard strategy. The default is None, where no timeout is set.
        timeout_rand_init:
            The maximum time in seconds to allow for conformer generation with random
            initialization. The default is None, where no timeout is set.
    Returns:
        mol:
            The molecule for which the 3D coordinates should be computed.
        conformer ID:
            The ID of the conformer that was generated.
        strategy:
            The strategy that was used for conformer generation. Either "default" or
            "random_init".
    """
    smiles = Chem.MolToSmiles(mol)

    # Try standard ETKDGv3 strategy first
    try:
        mol, conf_id = compute_conformer(
            mol,
            use_random_coord_init=False,
            remove_hs=remove_hs,
            timeout=timeout_standard,
        )
    except (ConformerGenerationError, FunctionTimedOut) as e:
        logger.warning(
            f"Exception when trying standard conformer generation for {smiles}: {e}, "
            + "trying random initialization"
        )

        # Try random coordinates as fallback
        try:
            mol, conf_id = compute_conformer(
                mol,
                use_random_coord_init=True,
                remove_hs=remove_hs,
                timeout=timeout_rand_init,
            )
        except (ConformerGenerationError, FunctionTimedOut) as e:
            logger.warning(
                "Exception when trying conformer generation with random "
                + f"initialization for {smiles}: {e}"
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
    # TODO: Expose timeouts as arguments
    # Test if conformer generation is possible
    try:
        mol, conf_id, strategy = multistrategy_compute_conformer(
            mol, remove_hs=True, timeout_standard=300, timeout_rand_init=300
        )
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


def get_name_match_argsort(
    atom_names: np.ndarray[str], ref_atom_names: np.ndarray[str]
) -> np.ndarray[int]:
    """Gets a sorting order for atom names based on a reference order.

    Args:
        atom_names:
            The current atom names.
        ref_atom_names:
            The reference atom names to sort by.

    Returns:
        The sorting order for the atom names to match the reference order. Any atom
        names not in the reference order are placed at the end.
    """
    # Map atom names to indices
    ref_order_map = {name: idx for idx, name in enumerate(ref_atom_names)}

    # Map the atom names in the molecule to the reference order (setting names that are
    # not in the reference to the end)
    atom_names_sort_keys = np.array(
        [ref_order_map.get(name, float("inf")) for name in atom_names]
    )

    # Sort the atoms by the reference order
    atom_names_new_order = np.argsort(atom_names_sort_keys)

    return atom_names_new_order


def get_cropped_permutations(
    mol: Mol,
    in_gt_mask: np.ndarray,
    in_crop_mask: np.ndarray,
    max_permutations: int = 1_000,
) -> np.ndarray:
    """Get the subset of symmetry-equivalent atom permutations matching crop and GT.

    This function computes the symmetry-equivalent atom permutations for a conformer
    using RDKit's `GetSubstructMatches` function. It then restricts these permutations
    so that the "slots" atoms can map to only correspond to atoms in the crop, and the
    indices atoms can be chosen from only correspond to atoms present in the
    ground-truth.

    Args:
        mol:
            The molecule for which the permutations should be computed.
        in_gt_mask:
            A boolean mask of atoms in the ground-truth structure.
        in_crop_mask:
            A boolean mask of atoms in the crop.
        max_permutations:
            The maximum number of permutations to compute.

    Returns:
        The symmetry-equivalent atom permutations that are valid for the crop and
        ground-truth.

        Shape: [n_permutations, n_atoms_in_crop]
    """
    # Define a mapping from the atom indices in the full conformer object to the atom
    # indices in the ground-truth
    conf_to_gt_index = np.full(len(in_gt_mask), -1, dtype=int)
    conf_to_gt_index[in_gt_mask] = np.arange(np.sum(in_gt_mask))

    # Get symmetry-equivalent atom permutations for this conformer following AF3 SI 4.2
    # (uses useChirality=False because that's also what RDKit's symmetry-corrected RMSD
    # uses)
    permutations = np.array(
        mol.GetSubstructMatches(
            mol, uniquify=False, maxMatches=max_permutations, useChirality=False
        )
    )

    # Map the permutations of full conformer atom indices to the ground-truth atoms
    gt_permutations = conf_to_gt_index[permutations]

    # Restrict permutations to atoms in the crop
    gt_permutations = gt_permutations[:, in_crop_mask]

    # Filter permutations that use atoms that are not in the ground-truth atoms
    gt_permutations = gt_permutations[np.all(gt_permutations != -1, axis=1)]

    assert gt_permutations.shape[1] == np.sum(in_crop_mask)
    assert gt_permutations.shape[0] >= 1

    return gt_permutations


def renumber_permutations(
    permutation: np.ndarray, required_gt_atoms: Iterable[int]
) -> np.ndarray:
    """Renumber permutation indices to reflect subsetted ground-truth atom indices.

    This function should be called after the ground-truth structure has been within
    `separate_cropped_and_gt`. In this case, there are now potentially fewer atoms in
    the ground-truth, so the positional indices of the permutations become outdated.
    This function takes all the required GT-atoms that are present in the new
    ground-truth, and renumbers all the permutation indices to reflect the new atom
    indices.

    Example:
    permutation: [[2, 4], [4, 2]]
    required_gt_atoms: [1, 2, 4, 5, 6] (superset of permutations because of other
                                        symmetry-equivalent molecules in structure)
    Result: [[2, 3], [3, 2]]

    Args:
        permutation:
            The permutation to renumber.
        required_gt_atoms:
            The set of ground-truth atoms that is still required in the new
            ground-truth.

    Returns:
        The renumbered permutation.
    """
    # Renumber full set of required GT atoms monotonically and define mapping from old
    # IDs
    required_gt_atoms = np.array(sorted(required_gt_atoms))
    required_gt_atoms_remapped = np.arange(len(required_gt_atoms))
    atom_idx_map = dict(zip(required_gt_atoms, required_gt_atoms_remapped))
    atom_idx_mapper = np.vectorize(lambda x: atom_idx_map[x])

    # Update the set of atoms in the permutations to reflect the new indices
    renumbered_permutation = atom_idx_mapper(permutation)

    return renumbered_permutation
