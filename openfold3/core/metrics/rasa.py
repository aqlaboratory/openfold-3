import logging

import biotite.structure as struc
import numpy as np
import torch

from openfold3.core.data.resources.residues import RESIDUE_SASA_SCALES


def _calculate_atom_sasa(chain, vdw_radii="ProtOr"):
    """
    Calculate the solvent accessible surface area (SASA) at the atom level.

    Parameters
    ----------
    chain : ndarray
        A chain array representing a single protein chain.
    vdw_radii : str or dict, optional
        The set of van der Waals radii to use for SASA calculation.
        Defaults to 'ProtOr'.

    Returns
    -------
    approx_atom_sasa : ndarray
        Per-atom SASA values.
    """
    return struc.sasa(chain, vdw_radii=vdw_radii)


def _calculate_residue_sasa(chain, approx_atom_sasa):
    """
    Aggregate atom-level SASA to residue-level SASA by summing
    all atom SASA values for each residue.

    Parameters
    ----------
    chain : ndarray
        A chain array representing a single protein chain.
    approx_atom_sasa : ndarray
        Per-atom SASA values.

    Returns
    -------
    approx_res_sasa : ndarray
        Per-residue SASA values.
    """
    return struc.apply_residue_wise(chain, approx_atom_sasa, np.sum)


def _identify_unresolved_residues(chain):
    """
    Identify unresolved residues based on atom occupancies.
    Residues with product of occupancies < 1 are considered unresolved.

    Parameters
    ----------
    chain : ndarray
        A chain array representing a single protein chain.

    Returns
    -------
    unresolved_residues : ndarray (dtype=bool)
        A boolean array indicating which residues are unresolved.
        True = unresolved, False = resolved.
    """
    return np.invert(
        struc.apply_residue_wise(chain, chain.occupancy, np.sum).astype(bool)
    )


def _map_residues_to_max_acc(chain, max_acc_dict, default_max_acc):
    """
    Map each residue to its maximum accessible surface area (max_acc).
    If a residue name is not in max_acc_dict, use default_max_acc.

    Parameters
    ----------
    chain : ndarray
        A chain array representing a single protein chain.
    max_acc_dict : dict
        Dictionary mapping residue names to their maximum accessible surface area.
    default_max_acc : float
        Default maximum accessible surface area for residues not in the dictionary.

    Returns
    -------
    max_acc : ndarray
        An array of maximum accessible surface area values, one per residue.
    """
    _, res_names = struc.get_residues(chain)
    return np.array(
        [max_acc_dict.get(res_name, default_max_acc) for res_name in res_names]
    )


def _compute_rasa(approx_res_sasa, max_acc):
    """
    Compute the Relative Accessible Surface Area (RASA) by dividing
    each residue's SASA by its maximum possible SASA, then clip values to [0, 1].

    Parameters
    ----------
    approx_res_sasa : ndarray
        Per-residue SASA values.
    max_acc : ndarray
        The maximum accessible surface area for each residue.

    Returns
    -------
    res_rasa : ndarray
        An array of RASA values for each residue, clipped to [0, 1].
    """
    res_rasa = approx_res_sasa / max_acc
    return np.clip(res_rasa, 0, 1)


def _smooth_rasa(res_rasa, window):
    """
    Smooth the RASA values using a simple moving average.

    Parameters
    ----------
    res_rasa : ndarray
        Raw per-residue RASA values.
    window : int
        The window size for the moving average.

    Returns
    -------
    smoothed_rasa : ndarray
        Smoothed per-residue RASA values.
    """
    half_w = (window - 1) // 2
    # Reflect padding helps avoid edge effects
    padded_rasa = np.pad(res_rasa, (half_w, half_w), mode="reflect")
    # Simple moving average
    smoothed_rasa = np.convolve(padded_rasa, np.ones(window), mode="valid") / window
    return smoothed_rasa


def calculate_res_rasa(
    chain, window, max_acc_dict=None, default_max_acc=113.0, vdw_radii="ProtOr"
):
    """
    Calculate per-residue Relative Accessible Surface Area (RASA) for a chain,
    using the method described in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767/.
    Adapted from:
    https://github.com/BioComputingUP/AlphaFold-disorder/blob/main/alphafold_disorder.py

    This function computes the unresolved residue RASA values needed for:
    1) Model selection
    2) Protein disorder score computation for sample ranking

    Parameters
    ----------
    chain : ndarray
        A chain array representing a single protein chain.
    window : int
        The window size for smoothing RASA values.
    max_acc_dict : dict, optional
        Dictionary mapping residue names to their maximum accessible surface area.
        If not provided, a default dictionary may be used
        (e.g., MAX_ACCESSIBLE_SURFACE_AREA).
    default_max_acc : float, optional
        Default max accessible surface area for residues not in the provided dictionary.
        Defaults to 113.0 (typical for ALA, but can be changed as needed).
    vdw_radii : str or dict, optional
        The set of van der Waals radii to use for SASA calculation.
        Defaults to 'ProtOr'.

    Returns
    -------
    smoothed_rasa : ndarray
        Smoothed per-residue RASA values.
    unresolved_residues : ndarray (dtype=bool)
        A boolean array indicating which residues are unresolved.

    Examples
    --------
    >>> chain = ...  # Some chain structure
    >>> smoothed_rasa, unresolved_residues = calculate_res_rasa(chain, window=25)
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
    struct_array,
    pol_type="peptide",
    window=25,
    max_acc_dict=None,
    default_max_acc=113.0,
    vdw_radii="ProtOr",
    residue_sasa_scale=None,
    pdb_id=None,
):
    """
    Process protein chains in a Biotite structure array and compute the average
    RASA value for all unresolved residues across all chains.

    Parameters
    ----------
    struct_array : ndarray
        The full structure array (which may contain multiple chains).
    pol_type : str, optional
        Polymer type to filter (default is "peptide").
    window : int, optional
        The window size for smoothing RASA values (default = 25).
    max_acc_dict : dict, optional
        Dictionary mapping residue names to their maximum accessible surface area.
    default_max_acc : float, optional
        Default maximum accessible surface area for residues not in the dictionary.
        Defaults to 113.0.
    vdw_radii : str or dict, optional
        The set of van der Waals radii to use for SASA calculation (default = "ProtOr").
    residue_sasa_scale : str, optional
        The residue SASA scale to use (default is "Sander").

    Returns
    -------
    float
        The mean RASA value for unresolved residues across all processed protein chains.
        Returns 0.0 if no unresolved residues are found or if an error occurs.

    Notes
    -----
    If any chain in the structure fails during RASA computation, a warning is logged
    and 0.0 is returned.

    Examples
    --------
    >>> array = ...  # A Biotite structure array with multiple chains
    >>> average_unresolved_rasa = process_proteins(array, pol_type="peptide", window=25)
    >>> print(average_unresolved_rasa)
    """
    if residue_sasa_scale is None:
        residue_sasa_scale = "Sander"

    if max_acc_dict is None:
        max_acc_dict = RESIDUE_SASA_SCALES[residue_sasa_scale]

    unresolved_residues_rasa = []

    # Filter the structure to only consider the specified polymer type (e.g., peptides)
    filtered = struct_array[struc.filter_polymer(struct_array, pol_type=pol_type)]

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
        return 0.0

    return np.mean(unresolved_residues_rasa)


def compute_rasa_batch(
    batch,
    outputs,
    pol_type="peptide",
    window=25,
    max_acc_dict=None,
    default_max_acc=113.0,
    vdw_radii="ProtOr",
    residue_sasa_scale=None,
):
    """
    Compute the average RASA value for unresolved residues across all protein chains
    in a batch of Biotite structure arrays.

    Parameters
    ----------
    batch : dict
        A batch of data containing Biotite structure arrays and other metadata.
    outputs : dict
        The model outputs containing predicted atom positions.
    pol_type : str, optional
        Polymer type to filter (default is "peptide").
    window : int, optional
        The window size for smoothing RASA values (default = 25).
    max_acc_dict : dict, optional
        Dictionary mapping residue names to their maximum accessible surface area.
    default_max_acc : float, optional
        Default maximum accessible surface area for residues not in the dictionary.
        Defaults to 113.0.
    vdw_radii : str or dict, optional
        The set of van der Waals radii to use for SASA calculation (default = "ProtOr").
    residue_sasa_scale : str, optional
        The residue SASA scale to use (default is "Sander").

    Returns
    -------
    list[float]
        The mean RASA value for unresolved residues across all processed protein chains
        in each structure array. Returns 0.0 if no unresolved residues are found
        or if an error

    Notes
    -----
    If any chain in the structure fails during RASA computation, a warning is logged
    and 0.0 is returned.
    """
    struct_arrays = batch["atom_array"]
    pdb_ids = batch["pdb_id"]
    N_batch, N_samples = outputs["atom_positions_predicted"].shape[
        :2
    ]  # (N_batch, N_samples, N_atoms, 3)
    unresolved_rasas = torch.zeros(
        (N_batch, N_samples), device=outputs["atom_positions_predicted"].device
    )
    for k, atom_arr in enumerate(struct_arrays):
        for sample in range(N_samples):
            atom_positions = outputs["atom_positions_predicted"][
                k, sample, : len(atom_arr), :3
            ]
            atom_arr.coord = atom_positions.detach().cpu().numpy()

            unresolved_rasas[k, sample] = process_proteins(
                atom_arr,
                pol_type=pol_type,
                window=window,
                max_acc_dict=max_acc_dict,
                default_max_acc=default_max_acc,
                vdw_radii=vdw_radii,
                residue_sasa_scale=residue_sasa_scale,
                pdb_id=pdb_ids[k],
            )
    return unresolved_rasas
