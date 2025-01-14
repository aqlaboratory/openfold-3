import logging

import biotite.structure as struc
import numpy as np
import torch

from openfold3.core.data.resources.residues import RESIDUE_SASA_SCALES, MoleculeType

logger = logging.getLogger(__name__)


def _calculate_atom_sasa(chain: np.ndarray, vdw_radii: str = "ProtOr") -> np.ndarray:
    """
    Calculate the solvent accessible surface area (SASA) at the atom level.

    Args:
        chain:
            A chain array representing a single protein chain.
        vdw_radii:
            The set of van der Waals radii to use for SASA calculation.
            Defaults to 'ProtOr'.

    Returns:
        approx_atom_sasa:
            Per-atom SASA values.
    """
    return struc.sasa(chain, vdw_radii=vdw_radii)


def _calculate_residue_sasa(
    chain: np.ndarray, approx_atom_sasa: np.ndarray
) -> np.ndarray:
    """
    Aggregate atom-level SASA to residue-level SASA by summing
    all atom SASA values for each residue.

    Args:
        chain:
            A chain array representing a single protein chain.
        approx_atom_sasa:
            Per-atom SASA values.

    Returns:
        approx_res_sasa:
            Per-residue SASA values.
    """
    return struc.apply_residue_wise(chain, approx_atom_sasa, np.sum)


def _identify_unresolved_residues(
    chain: np.ndarray,
) -> np.ndarray:
    """
    Identify unresolved residues based on atom occupancies.
    Residues with sum of resolved atoms < 1 are considered unresolved.

    Args:
        chain:
            A chain array representing a single protein chain.
    Returns:
        unresolved_residues:
            A boolean array indicating which residues are unresolved.
            True = unresolved, False = resolved.
    """

    return np.invert(
        struc.apply_residue_wise(chain, chain.atom_resolved_mask, np.sum).astype(bool)
    )


def _map_residues_to_max_acc(
    chain: np.ndarray, max_acc_dict: dict, default_max_acc: float
) -> np.ndarray:
    """
    Map each residue to its maximum accessible surface area (max_acc).
    If a residue name is not in max_acc_dict, use default_max_acc.

    Args:
        chain:
            A chain array representing a single protein chain.
        max_acc_dict:
            Dictionary mapping residue names to their maximum accessible surface area.
        default_max_acc:
            Default maximum accessible surface area for residues not in the dictionary.

    Returns:
        max_acc:
            An array of maximum accessible surface area values, one per residue.
    """
    _, res_names = struc.get_residues(chain)
    return np.array(
        [max_acc_dict.get(res_name, default_max_acc) for res_name in res_names]
    )


def _compute_rasa(approx_res_sasa: np.ndarray, max_acc: np.ndarray) -> np.ndarray:
    """
    Compute the Relative Accessible Surface Area (RASA) by dividing
    each residue's SASA by its maximum possible SASA, then clip values to [0, 1].

    Args:
        approx_res_sasa:
            Per-residue SASA values.
        max_acc:
            The maximum accessible surface area for each residue.

    Returns:
        res_rasa:
            An array of RASA values for each residue, clipped to [0, 1].
    """
    res_rasa = approx_res_sasa / max_acc
    return np.clip(res_rasa, 0, 1)


def _smooth_rasa(res_rasa: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth the RASA values using a simple moving average.

    Args:
        res_rasa:
            Raw per-residue RASA values.
        window:
            The window size for the moving average.

    Returns:
        smoothed_rasa:
            Smoothed per-residue RASA values.
    """
    half_w = (window - 1) // 2
    # Reflect padding helps avoid edge effects
    padded_rasa = np.pad(res_rasa, (half_w, half_w), mode="reflect")
    # Simple moving average
    smoothed_rasa = np.convolve(padded_rasa, np.ones(window), mode="valid") / window
    return smoothed_rasa


def calculate_res_rasa(
    chain: np.ndarray,
    window: int,
    max_acc_dict: dict = None,
    default_max_acc: float = 113.0,
    vdw_radii: str = "ProtOr",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-residue Relative Accessible Surface Area (RASA) for a chain,
    using the method described in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767/.
    Adapted from:
    https://github.com/BioComputingUP/AlphaFold-disorder/blob/main/alphafold_disorder.py

    This function computes the unresolved residue RASA values needed for:
    1) Model selection
    2) Protein disorder score computation for sample ranking

    Args:
        chain:
            A chain array representing a single protein chain.
        window:
            The window size for smoothing RASA values.
        max_acc_dict:
            Dictionary mapping residue names to their maximum accessible surface area.
            If not provided, a default dictionary may be used
            (e.g., MAX_ACCESSIBLE_SURFACE_AREA).
        default_max_acc:
            Default max accessible surface area for residues not in the provided
                dictionary.
            Defaults to 113.0 (typical for ALA, but can be changed as needed).
        vdw_radii:
            The set of van der Waals radii to use for SASA calculation.
            Defaults to 'ProtOr'.

    Returns:
        smoothed_rasa:
            Smoothed per-residue RASA values.
        unresolved_residues:
            A boolean array indicating which residues are unresolved.
    """
    if max_acc_dict is None:
        max_acc_dict = RESIDUE_SASA_SCALES

    # 1. Calculate SASA at the atom level
    approx_atom_sasa = _calculate_atom_sasa(chain, vdw_radii=vdw_radii)

    # 2. Aggregate SASA to residue level
    approx_res_sasa = _calculate_residue_sasa(chain, approx_atom_sasa)

    # 3. Identify unresolved residues
    unresolved_residues = _identify_unresolved_residues(chain)

    # 4. Map each residue to its max accessible surface area
    max_acc = _map_residues_to_max_acc(chain, max_acc_dict, default_max_acc)

    # 5. Compute RASA (clip to [0, 1])
    res_rasa = _compute_rasa(approx_res_sasa, max_acc)

    # 6. Apply smoothing
    smoothed_rasa = _smooth_rasa(res_rasa, window)

    return smoothed_rasa, unresolved_residues


def process_proteins(
    struct_array: np.ndarray,
    window: int = 25,
    max_acc_dict: dict = None,
    default_max_acc: float = 113.0,
    vdw_radii: str = "ProtOr",
    residue_sasa_scale: str = None,
    pdb_id: str = None,
) -> float:
    """
    Process protein chains in a Biotite structure array and compute the average
    RASA value for all unresolved residues across all chains.

    Args:
        struct_array:
            The full structure array (which may contain multiple chains).
        window:
            The window size for smoothing RASA values (default = 25).
        max_acc_dict:
            Dictionary mapping residue names to their maximum accessible surface area.
        default_max_acc:
            Default maximum accessible surface area for residues not in the dictionary.
            Defaults to 113.0.
        vdw_radii:
            The set of van der Waals radii to use for SASA calculation
                (default = "ProtOr").
        residue_sasa_scale:
            The residue SASA scale to use (default is "Sander").

    Returns:
        The mean RASA value for unresolved residues across all processed protein chains.
        Returns NaN if no unresolved residues are found or if an error occurs.

    Notes:
        If any chain in the structure fails during RASA computation, a warning is logged
        and nan is returned.
    """
    if residue_sasa_scale is None:
        residue_sasa_scale = "Sander"

    if max_acc_dict is None:
        max_acc_dict = RESIDUE_SASA_SCALES[residue_sasa_scale]

    unresolved_residues_rasa = []

    # Filter the structure to only consider the specified polymer type (e.g., peptides)
    filtered = struct_array[struct_array.molecule_type_id == MoleculeType.PROTEIN]

    if len(filtered) == 0:
        logging.warning(f"No protein chains found in pdb_id={pdb_id}")
        return float("nan")

    # Set a default max_acc for fallback residues
    if default_max_acc is None:
        default_max_acc = max_acc_dict.get("ALA", 113.0)

    for chain in struc.chain_iter(filtered):
        try:
            res_rasa, unresolved_residues = calculate_res_rasa(
                chain=chain,
                window=window,
                max_acc_dict=max_acc_dict,
                default_max_acc=default_max_acc,
                vdw_radii=vdw_radii,
            )
            # Extend the list with RASA values for all unresolved residues in this chain
            unresolved_residues_rasa.extend(res_rasa[unresolved_residues])
        except Exception as e:
            logging.warning(f"RASA computation failed for pdb_id={pdb_id}: {e}")
    if not unresolved_residues_rasa:
        return float("nan")

    return np.mean(unresolved_residues_rasa)


def compute_rasa_batch(
    batch: dict,
    outputs: dict,
    window: int = 25,
    max_acc_dict: dict = None,
    default_max_acc: float = 113.0,
    vdw_radii: str = "ProtOr",
    residue_sasa_scale: str = None,
) -> torch.Tensor:
    """
    Compute the average RASA value for unresolved residues across all protein chains
    in a batch of Biotite structure arrays.

    Args:
        batch :
            A batch of data containing Biotite structure arrays and other metadata.
        outputs:
            The model outputs containing predicted atom positions.
        window :
            The window size for smoothing RASA values (default = 25).
        max_acc_dict:
            Dictionary mapping residue names to their maximum accessible surface area.
        default_max_acc:
            Default maximum accessible surface area for residues not in the dictionary.
            Defaults to 113.0.
        vdw_radii:
            The set of van der Waals radii to use for SASA calculation
                (default = "ProtOr").
        residue_sasa_scale:
            The residue SASA scale to use (default is "Sander").

    Returns:
        The mean RASA value for unresolved residues across all processed protein chains
        in each structure array. Returns 0.0 if no unresolved residues are found
        or if an error occurs.

    Notes:
        If any chain in the structure fails during RASA computation, a warning is logged
        and NaN is returned.
    """
    struct_arrays = batch["atom_array"]
    pdb_ids = batch["pdb_id"]
    n_batch, n_samples = outputs["atom_positions_predicted"].shape[
        :2
    ]  # (N_batch, N_samples, N_atoms, 3)
    unresolved_rasas = torch.zeros(
        (n_batch, n_samples), device=outputs["atom_positions_predicted"].device
    )
    for k, atom_arr in enumerate(struct_arrays):
        atom_arr.set_annotation(
            "atom_resolved_mask", np.ones_like(atom_arr.occupancy, dtype=bool)
        )
        for sample in range(n_samples):
            atom_positions = outputs["atom_positions_predicted"][
                k, sample, : len(atom_arr), :3
            ]
            resolved_mask = (
                batch["ground_truth"]["atom_resolved_mask"][k, sample, : len(atom_arr)]
                .detach()
                .cpu()
                .numpy()
            )
            atom_arr.coord = atom_positions.detach().cpu().numpy()
            atom_arr.atom_resolved_mask = resolved_mask

            unresolved_rasas[k, sample] = process_proteins(
                struct_array=atom_arr,
                window=window,
                max_acc_dict=max_acc_dict,
                default_max_acc=default_max_acc,
                vdw_radii=vdw_radii,
                residue_sasa_scale=residue_sasa_scale,
                pdb_id=pdb_ids[k],
            )
    return unresolved_rasas
