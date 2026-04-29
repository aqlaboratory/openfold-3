# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ConformerStrategy:
    """A single conformer-generation strategy.

    The `name` is persisted as `conformer_gen_strategy` in the preprocessing cache
    (see `PreprocessingReferenceMoleculeData`); do not rename existing entries.
    """

    name: str
    kwargs: dict = field(default_factory=dict)
    default_timeout: float | None = None


CONFORMER_STRATEGIES: tuple[ConformerStrategy, ...] = (
    ConformerStrategy("default", {"use_random_coord_init": False}),
    ConformerStrategy("small_ring_torsions", {"use_small_ring_torsions": True}),
    ConformerStrategy("random_init", {"use_random_coord_init": True}),
)


@dataclass(frozen=True)
class ConformerResult:
    mol: Mol
    conf_id: int
    strategy: str


def _get_strategy(name: str) -> ConformerStrategy:
    strat = next((s for s in CONFORMER_STRATEGIES if s.name == name), None)
    if strat is None:
        raise ValueError(f"Unknown conformer strategy: {name!r}")
    return strat


def _compute_conformer(
    mol: Mol,
    use_random_coord_init: bool = False,
    use_small_ring_torsions: bool = False,
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
        use_small_ring_torsions:
            Whether to enable RDKit's small-ring-torsion potentials. Helpful for
            macrocycles and fused-ring systems (e.g. porphyrins / heme) where the
            default ETKDGv3 torsion preferences fail to embed.
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

    if use_small_ring_torsions:
        strategy.useSmallRingTorsions = True

    strategy.clearConfs = False
    # RDKit always seems to start from some internal seed instead of a truly random seed
    # initialization if no seed is given, so we set a random seed here
    strategy.randomSeed = random.randint(0, 10**9)

    # Disable overly verbose conformer generation warnings
    with rdBase.BlockLogs():
        if timeout:
            conf_id = func_timeout(
                timeout=timeout, func=AllChem.EmbedMolecule, args=(mol, strategy)
            )
        else:
            conf_id = AllChem.EmbedMolecule(mol, strategy)

    if remove_hs:
        mol = safe_remove_all_hs(mol)

    if conf_id == -1:
        raise ConformerGenerationError("Failed to generate 3D coordinates")

    return mol, conf_id


# TODO: could improve warning handling of this to send less UFFTYPER warnings
def multistrategy_compute_conformer(
    mol: Mol,
    *,
    remove_hs: bool = True,
    start_from: str = "default",
    timeouts: dict[str, float | None] | None = None,
) -> ConformerResult:
    """Computes 3D coordinates for a molecule trying different strategies in order.

    Iterates over `CONFORMER_STRATEGIES` starting at `start_from`, returning the first
    strategy that succeeds. If every strategy from `start_from` onward fails or times
    out, a `ConformerGenerationError` is raised.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
        remove_hs:
            Whether to remove hydrogens from the molecule after conformer generation.
        start_from:
            The name of the strategy at which to begin the chain. Strategies that come
            before it in `CONFORMER_STRATEGIES` are skipped. Defaults to the first
            strategy ("default").
        timeouts:
            Optional per-strategy timeout overrides keyed by strategy name. A value of
            `None` disables the timeout for that strategy. Strategies with no entry
            here fall back to their `default_timeout`.

    Returns:
        A `ConformerResult` carrying the molecule, the generated conformer's id, and
        the name of the strategy that succeeded.
    """
    timeouts = timeouts or {}
    smiles = Chem.MolToSmiles(mol)

    # Validate `start_from` and locate it in the chain.
    _get_strategy(start_from)
    iter_start = next(
        i for i, s in enumerate(CONFORMER_STRATEGIES) if s.name == start_from
    )

    last_exc: Exception | None = None
    for strat in CONFORMER_STRATEGIES[iter_start:]:
        try:
            mol_out, conf_id = _compute_conformer(
                mol,
                remove_hs=remove_hs,
                timeout=timeouts.get(strat.name, strat.default_timeout),
                **strat.kwargs,
            )
        except (ConformerGenerationError, FunctionTimedOut) as e:
            logger.warning(
                f"Conformer strategy {strat.name!r} failed for {smiles}: {e}"
            )
            last_exc = e
            continue
        return ConformerResult(mol=mol_out, conf_id=conf_id, strategy=strat.name)

    raise ConformerGenerationError("Failed to generate 3D coordinates") from last_exc


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
        result = multistrategy_compute_conformer(
            mol, remove_hs=True, timeouts={"default": 300, "random_init": 300}
        )
        mol = result.mol
        strategy = result.strategy
        conf = mol.GetConformer(result.conf_id)
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
    atom_idx_map = dict(zip(required_gt_atoms, required_gt_atoms_remapped, strict=True))
    atom_idx_mapper = np.vectorize(lambda x: atom_idx_map[x])

    # Update the set of atoms in the permutations to reflect the new indices
    renumbered_permutation = atom_idx_mapper(permutation)

    return renumbered_permutation
