"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure

logger = logging.getLogger(__name__)


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(self, output_dir, structure_format, full_confidence_out_format):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_out_format

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(atom_array, output_file, include_bonds=True)

    def write_confidence_scores(self, confidence_scores: dict, output_prefix: Path):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["predicted_distance_error"]
        gpde = confidence_scores["global_predicted_distance_error"]

        # Single-valued aggregated confidence scores
        aggregated_confidence_scores = {
            "avg_plddt": torch.mean(plddt).item(),
            "gpde": gpde,
        }
        out_file_agg = output_prefix / "confidences_aggregated.json"
        out_file_agg.write_text(json.dumps(aggregated_confidence_scores, indent=4))

        # Full confience scores
        full_confidence_scores = {"plddt": plddt, "pde": pde}
        out_fmt = self.full_confidence_format
        out_file_full = output_prefix / "confidences" / f".{out_fmt}"

        if out_fmt == "json":
            out_file_full.write_text(
                json.dumps(
                    full_confidence_scores, indent=4, default=lambda x: x.tolist()
                )
            )
        elif out_fmt == "npz":
            np.savez_compressed(out_file_full, **full_confidence_scores)

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

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]

            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            # Extract attributes for the current batch
            atom_array_batch = batch["atom_array"][b]
            predicted_coords_batch = (
                outputs["atom_positions_predicted"][b].cpu().float().numpy()
            )
            confidence_scores_batch = {
                key: value[b].cpu().float() if len(value.shape) > 1 else value
                for key, value in confidence_scores.items()
            }

            # Iterate over all diffusion samples
            for s in range(sample_size):
                file_prefix = output_subdir / f"{query_id}_seed_{seed}_sample_{s + 1}"

                confidence_scores_sample = {
                    key: value[s] if len(value.shape) > 1 else value.item()
                    for key, value in confidence_scores_batch.items()
                }

                predicted_coords_sample = predicted_coords_batch[s]

                # Save predicted structure
                structure_file = file_prefix / f"_model.{self.structure_format}"
                self.write_structure_prediction(
                    atom_array=atom_array_batch,
                    predicted_coords=predicted_coords_sample,
                    plddt=confidence_scores_sample["plddt"],
                    output_file=structure_file,
                )

                # Save confidence metrics
                self.write_confidence_scores(
                    confidence_scores=confidence_scores_sample,
                    file_prefix=file_prefix,
                )
