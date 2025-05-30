"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure


def write_confidence_scores(confidence_scores: dict, output_path: Path, sample: int):
    """Writes confidence scores to disk"""
    # Convert tensors to lists
    json_compatible_dict = {
        key: value[sample].tolist()
        if isinstance(value, torch.Tensor) and len(value.shape) > 1
        else value
        for key, value in confidence_scores.items()
    }
    write_json(json_compatible_dict, output_path)


def write_json(json_dict: dict, output_path: Path):
    """Writes a dictionary to disk in JSON format"""
    # only write if the file does not exist
    if not output_path.exists():
        with open(output_path, "w") as f:
            json.dump(json_dict, f)


def write_structure_prediction(
    seed: int,
    atom_array: structure.AtomArray,
    predicted_coords: np.ndarray,
    pdb_id: str,
    output_dir: Path,
    atom_unresolved_mask: np.ndarray = None,
):
    """Writes predicted coordinates to atom_array and writes mmcif file to disk"""
    status = "predicted"

    # Overwrite coordinates in atom_array
    for sample in range(predicted_coords.shape[0]):
        atom_array.coord = predicted_coords[sample]
        if atom_unresolved_mask is not None:
            atom_array.occupancy = atom_unresolved_mask[sample]
            atom_array_only_resolved = atom_array[atom_array.occupancy != 0]
            status = "ground_truth"

        output_path = (
            Path(output_dir)
            / pdb_id
            / f"model_{seed}"
            / f"{pdb_id}_{status}_seed_{seed}_sample_{sample + 1}.cif"
        )

        os.makedirs(output_path.parent, exist_ok=True)

        # Write the output file
        logging.info(f"Writing predicted structure for {pdb_id} to {output_path}")
        if status == "predicted":
            write_structure(atom_array, output_path, include_bonds=True)
        else:
            write_structure(atom_array_only_resolved, output_path, include_bonds=True)


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(self, output_dir):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        is_repeated_sample = batch.get("repeated_sample")
        if outputs is None or is_repeated_sample:
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        for i in range(len(batch["atom_array"])):
            seed = batch["seed"][i]
            pdb_id = batch["query_id"][i]

            atom_array = batch["atom_array"][i]
            predicted_coords = (
                outputs["atom_positions_predicted"][i].cpu().float().numpy()
            )

            confidence_scores_bs = {
                key: value[i].cpu().float() if len(value.shape) > 1 else value.item()
                for key, value in confidence_scores.items()
            }

            # TODO: UPDATE THIS WHEN WE HAVE THE CONFIDENCE SCORES: pTM, Sample Ranking
            # NOW ONLY KEEP PLDDT SCORES
            confidence_scores_sample = {"plddt": confidence_scores_bs["plddt"]}

            write_structure_prediction(
                seed,
                atom_array,
                predicted_coords,
                pdb_id,
                self.output_dir,
            )

            for sample in range(predicted_coords.shape[0]):
                confidence_filename = (
                    f"{pdb_id}_predicted_seed_{seed}_sample_"
                    f"{sample + 1}_confidence_scores.json"
                )
                output_confidence_path = (
                    Path(
                        self.output_dir,
                    )
                    / pdb_id
                    / f"model_{seed}"
                    / confidence_filename
                )
                write_confidence_scores(
                    confidence_scores_sample, output_confidence_path, sample
                )
